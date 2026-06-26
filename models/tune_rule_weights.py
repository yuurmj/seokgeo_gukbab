import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from build_fire_proxy import build_proxy

try:
    import optuna
except ImportError:
    optuna = None

# 규칙 기반 모델 가중치 보조 분석
# 1) 규칙 기반 결과 CSV 로드
# 2) 기상/공간/상호작용 위험 점수 정리
# 3) 산불 이력 proxy label 생성
# 4) Logistic Regression 기반 선형 방향성 진단
# 5) LightGBM + SHAP 기반 중요도 비교
# 6) Random Forest 기반 중요도 보조 검증
# 7) Optuna 기반 가중치/threshold 후보 탐색
# 8) 2024년 산불 proxy 기준 최종 검증

DEFAULT_RULE_CANDIDATES = [
    BASE_DIR / "outputs" / "final_risk_result_train.csv",
    BASE_DIR / "outputs" / "pole_spatial_risk_score_train.csv",
    BASE_DIR / "outputs" / "final_risk_result.csv",
    BASE_DIR / "outputs" / "pole_spatial_risk_score.csv",
]
DEFAULT_FIRE_INPUT = BASE_DIR / "data" / "fire_history.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "weight_tuning"

# X 컬럼: 규칙 기반 영역별 위험 점수
COMPONENT_COLUMNS = [
    "weather_risk_score",
    "spatial_risk_score",
    "interaction_risk_score",
]


def parse_years(value):
    """연도 문자열 → 정수 리스트 변환."""
    return [int(year.strip()) for year in value.split(",") if year.strip()]


def require_columns(data, columns, data_name):
    """필수 컬럼 확인."""
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{data_name}에 필요한 컬럼이 없습니다: {missing}")


def resolve_default_rule_input():
    """규칙 기반 결과 기본 파일 선택."""
    for path in DEFAULT_RULE_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_RULE_CANDIDATES[0]


def prepare_rule_scores(data):
    """규칙 기반 점수 정리."""
    require_columns(
        data,
        [
            "pole_id",
            "lon",
            "lat",
            "weather_risk_score",
            "spatial_risk_score",
        ],
        "규칙 기반 결과",
    )

    result = data.copy()
    if "interaction_risk_score" not in result.columns:
        result["interaction_risk_score"] = (
            pd.to_numeric(result["weather_risk_score"], errors="coerce")
            * pd.to_numeric(result["spatial_risk_score"], errors="coerce")
        )

    numeric_columns = [
        "weather_risk_score",
        "spatial_risk_score",
        "interaction_risk_score",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result[column].fillna(result[column].median())

    return result


def normalize_components(data):
    """영역별 점수 0~1 정규화."""
    normalized = data[COMPONENT_COLUMNS].copy()
    for column in COMPONENT_COLUMNS:
        min_value = normalized[column].min()
        max_value = normalized[column].max()
        if max_value == min_value:
            normalized[column] = 0.0
        else:
            normalized[column] = (
                (normalized[column] - min_value) / (max_value - min_value)
            )
    return normalized


def metric_dict(y_true, y_pred):
    """평가 지표 생성."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "positive_rate": float(np.mean(y_pred_array)),
        "selected_count": int(y_pred_array.sum()),
        "proxy_total": int(y_true_array.sum()),
        "captured_proxy_count": int(((y_true_array == 1) & (y_pred_array == 1)).sum()),
    }


def save_json(data, path):
    """JSON 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fit_logistic_weights(X, y, output_dir):
    """Logistic Regression 기반 선형 방향성 진단."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X, y)
    coef = pd.Series(model.coef_[0], index=COMPONENT_COLUMNS)

    # 양수 계수 기반 참고 가중치 생성
    positive_coef = coef.clip(lower=0)
    has_positive_coef = positive_coef.sum() > 0
    if has_positive_coef:
        diagnostic_weights = positive_coef / positive_coef.sum()
        weight_note = "positive_coefficients"
    else:
        diagnostic_weights = coef.abs() / coef.abs().sum()
        weight_note = "absolute_coefficients_for_diagnosis_only"

    result = pd.DataFrame(
        {
            "component": COMPONENT_COLUMNS,
            "logistic_coef": coef.values,
            "diagnostic_weight": diagnostic_weights.values,
            "coef_direction": np.where(coef.values >= 0, "positive", "negative"),
            "use_as_final_weight": has_positive_coef,
        }
    ).sort_values("diagnostic_weight", ascending=False)
    result.to_csv(output_dir / "logistic_weights.csv", index=False)
    summary = {
        "weight_note": weight_note,
        "has_positive_coefficient": bool(has_positive_coef),
        "all_coefficients_non_positive": bool((coef <= 0).all()),
    }
    return model, diagnostic_weights.to_dict(), summary


def fit_lightgbm(X_train, y_train, X_valid, y_valid, output_dir):
    """LightGBM + SHAP 기반 비선형 중요도 비교."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("LightGBM이 설치되어 있지 않아 중요도 분석을 건너뜁니다.")
        return None

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    importance = pd.DataFrame(
        {
            "feature": COMPONENT_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "lightgbm_feature_importance.csv", index=False)

    # 2024년 산불 proxy 기준 검증
    pred = model.predict(X_valid)
    save_json(metric_dict(y_valid, pred), output_dir / "lightgbm_valid_metrics.json")

    try:
        # SHAP 캐시 경로 설정
        matplotlib_cache = output_dir / "matplotlib_cache"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_valid)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_importance = pd.DataFrame(
            {
                "feature": COMPONENT_COLUMNS,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(output_dir / "lightgbm_shap_importance.csv", index=False)
    except ImportError:
        print("SHAP이 설치되어 있지 않아 LightGBM feature_importance만 저장했습니다.")

    return model


def fit_random_forest(X_train, y_train, X_valid, y_valid, output_dir):
    """Random Forest 기반 중요도 보조 검증."""
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    importance = pd.DataFrame(
        {
            "feature": COMPONENT_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "random_forest_feature_importance.csv", index=False)

    # 2024년 산불 proxy 기준 검증
    pred = model.predict(X_valid)
    save_json(metric_dict(y_valid, pred), output_dir / "random_forest_valid_metrics.json")

    return model


def make_weighted_score(X, weights):
    """가중합 위험점수 생성."""
    return sum(X[column] * weights[column] for column in COMPONENT_COLUMNS)


def best_top_rate_for_weights(X, y, weights, min_top_rate=0.01, max_top_rate=0.50):
    """주어진 가중치에서 F1이 최대가 되는 컷(top_rate)을 정확히 계산.

    점수를 내림차순으로 정렬한 뒤, "상위 k개를 1로 찍을 때"의 F1을
    k=1..n 전부에 대해 누적으로 구해 그중 최댓값을 고른다.
    5개 후보를 샘플링하는 대신, 가능한 모든 컷을 한 번에 평가하는 방식.
    """
    score = np.asarray(make_weighted_score(X, weights), dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    P = int(y.sum())

    # 양성이 하나도 없으면 컷을 정할 수 없으므로 중앙값으로 fallback
    if P == 0 or n == 0:
        return {
            "top_rate": 0.5,
            "threshold": float(np.median(score)) if n else 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    order = np.argsort(-score)          # 점수 내림차순 정렬
    y_sorted = y[order]
    score_sorted = score[order]

    tp = np.cumsum(y_sorted)            # 상위 k개 선택 시 누적 TP
    k = np.arange(1, n + 1)
    precision = tp / k
    recall = tp / P
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    # 운영상 의미가 없거나 불안정한 극단 컷은 후보에서 제외
    low_k = max(1, int(np.ceil(min_top_rate * n)))
    high_k = max(low_k, int(np.floor(max_top_rate * n)))
    mask = np.zeros(n, dtype=bool)
    mask[low_k - 1:high_k] = True
    f1_masked = np.where(mask, f1, -1.0)

    best_k = int(np.argmax(f1_masked)) + 1
    return {
        "top_rate": best_k / n,
        "threshold": float(score_sorted[best_k - 1]),
        "f1": float(f1[best_k - 1]),
        "precision": float(precision[best_k - 1]),
        "recall": float(recall[best_k - 1]),
    }


def evaluate_weight_candidate(X, y, weights, top_rate):
    """가중치/상위비율 후보 평가."""
    score = make_weighted_score(X, weights)
    threshold = score.quantile(1 - top_rate)
    pred = (score >= threshold).astype(int)
    metrics = metric_dict(y, pred)
    metrics["top_rate"] = top_rate
    metrics["threshold"] = float(threshold)
    return metrics


def random_search_weights(X_train, y_train, X_valid, y_valid, n_trials, output_dir):
    """Optuna 미설치 시 무작위 후보 탐색."""
    rng = np.random.default_rng(42)
    rows = []

    for trial in range(n_trials):
        sampled = rng.dirichlet(np.ones(len(COMPONENT_COLUMNS)))
        weights = dict(zip(COMPONENT_COLUMNS, sampled))
        # top_rate 그리드 루프 제거: 가중치별 최적 컷을 직접 계산
        train_cut = best_top_rate_for_weights(X_train, y_train, weights)
        train_metrics = evaluate_weight_candidate(
            X_train, y_train, weights, train_cut["top_rate"]
        )
        valid_cut = best_top_rate_for_weights(X_valid, y_valid, weights)
        valid_metrics = evaluate_weight_candidate(
            X_valid, y_valid, weights, valid_cut["top_rate"]
        )
        rows.append(
            {
                "trial": trial,
                "method": "random_search",
                **{f"weight_{key}": value for key, value in weights.items()},
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"valid_{key}": value for key, value in valid_metrics.items()},
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["train_f1", "train_recall", "valid_f1"], ascending=False
    )
    result.to_csv(output_dir / "weight_search_results.csv", index=False)
    return result.iloc[0].to_dict()


def optuna_search_weights(X_train, y_train, X_valid, y_valid, n_trials, output_dir):
    """Optuna 기반 가중치/threshold 후보 탐색."""
    if optuna is None:
        return random_search_weights(
            X_train, y_train, X_valid, y_valid, n_trials, output_dir
        )

    rows = []

    def objective(trial):
        # 가중치 합 1 정규화
        raw_weights = np.array(
            [
                trial.suggest_float(column, 0.0, 1.0)
                for column in COMPONENT_COLUMNS
            ]
        )
        if raw_weights.sum() == 0:
            weights_array = np.ones(len(COMPONENT_COLUMNS)) / len(COMPONENT_COLUMNS)
        else:
            weights_array = raw_weights / raw_weights.sum()
        weights = dict(zip(COMPONENT_COLUMNS, weights_array))

        # top_rate는 더 이상 샘플하지 않는다.
        # 가중치가 정해지면 컷은 정답이 있는 계산 문제이므로 직접 최적값을 구한다.
        # 가중치 선택은 train(4개 연도, 더 안정적) 기준으로 한다.
        train_cut = best_top_rate_for_weights(X_train, y_train, weights)
        train_metrics = evaluate_weight_candidate(
            X_train, y_train, weights, train_cut["top_rate"]
        )
        # valid 성능은 valid 자체의 최적 컷에서 측정해 일반화 정도를 함께 기록
        valid_cut = best_top_rate_for_weights(X_valid, y_valid, weights)
        valid_metrics = evaluate_weight_candidate(
            X_valid, y_valid, weights, valid_cut["top_rate"]
        )
        rows.append(
            {
                "trial": trial.number,
                "method": "optuna",
                **{f"weight_{key}": value for key, value in weights.items()},
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"valid_{key}": value for key, value in valid_metrics.items()},
            }
        )
        return train_metrics["f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    result = pd.DataFrame(rows).sort_values(
        ["train_f1", "train_recall", "valid_f1"], ascending=False
    )
    result.to_csv(output_dir / "weight_search_results.csv", index=False)
    return result.iloc[0].to_dict()


def parse_args():
    """CLI 옵션 정의."""
    parser = argparse.ArgumentParser(
        description="Tune rule-based risk weights with fire proxy labels."
    )
    parser.add_argument("--rule-input", default=str(resolve_default_rule_input()))
    parser.add_argument("--fire-input", default=str(DEFAULT_FIRE_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--radius-km", type=float, default=3.0)
    parser.add_argument("--train-years", default="2020,2021,2022,2023")
    parser.add_argument("--valid-years", default="2024")
    parser.add_argument("--n-trials", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))

    # 입력 파일 확인
    rule_path = Path(args.rule_input)
    fire_path = Path(args.fire_input)
    if not rule_path.exists():
        raise FileNotFoundError(
            f"규칙 기반 결과 파일이 없습니다: {rule_path}\n"
            "먼저 models/rule_based_model.py를 실행해 final_risk_result_train.csv "
            "또는 pole_spatial_risk_score_train.csv를 생성하세요."
        )
    if not fire_path.exists():
        raise FileNotFoundError(f"산불 이력 파일이 없습니다: {fire_path}")

    # 규칙 기반 결과/산불 이력 로드
    rule_data = pd.read_csv(rule_path, encoding="utf-8-sig")
    fire_data = pd.read_csv(fire_path, encoding="utf-8-sig")
    fire_data["occu_year"] = pd.to_numeric(
        fire_data["occu_year"], errors="coerce"
    ).astype("Int64")

    train_years = parse_years(args.train_years)
    valid_years = parse_years(args.valid_years)

    # X 후보 점수 생성
    data = prepare_rule_scores(rule_data)

    # 학습용 산불 proxy 생성
    data = build_proxy(
        data,
        fire_data,
        radius_km=args.radius_km,
        years=train_years,
        prefix="train",
    )
    # 검증용 산불 proxy 생성
    data = build_proxy(
        data,
        fire_data,
        radius_km=args.radius_km,
        years=valid_years,
        prefix="valid",
    )

    train_label = f"train_fire_proxy_{int(args.radius_km)}km"
    valid_label = f"valid_fire_proxy_{int(args.radius_km)}km"

    # 중간 데이터셋 저장
    data.to_csv(output_dir / "proxy_component_dataset.csv", index=False)

    # X/y 생성
    X = normalize_components(data)
    y_train = data[train_label].astype(int)
    y_valid = data[valid_label].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "학습용 proxy label이 한 종류뿐입니다. 반경 또는 학습 연도를 조정하세요."
        )

    # Logistic Regression 기반 선형 방향성 진단
    logistic_model, logistic_weights, logistic_diagnostic = fit_logistic_weights(
        X, y_train, output_dir
    )
    logistic_pred = logistic_model.predict(X)
    save_json(
        {
            "train_years": train_years,
            "valid_years": valid_years,
            "radius_km": args.radius_km,
            "train_proxy_rate": float(y_train.mean()),
            "valid_proxy_rate": float(y_valid.mean()),
            "logistic_diagnostic_weights": logistic_weights,
            "logistic_diagnostic": logistic_diagnostic,
            "logistic_train_metrics": metric_dict(y_train, logistic_pred),
            "logistic_valid_metrics": metric_dict(y_valid, logistic_pred),
        },
        output_dir / "logistic_summary.json",
    )

    # LightGBM + SHAP 기반 중요도 비교
    fit_lightgbm(X, y_train, X, y_valid, output_dir)

    # Random Forest 기반 중요도 보조 검증
    fit_random_forest(X, y_train, X, y_valid, output_dir)

    # Optuna 기반 가중치 탐색 (컷은 가중치별 최적값을 직접 계산)
    best = optuna_search_weights(
        X, y_train, X, y_valid, args.n_trials, output_dir
    )

    # 최종 컷(0/1 기준)은 검증연도(2024)에서 F1 최대가 되는 지점으로 정한다.
    # 가중치는 train(4개 연도)으로 안정적으로 고르고, 0/1 경계선만
    # test에 가장 가까운 미래 데이터(valid)에서 맞춘다.
    best_weights = {
        column: float(best[f"weight_{column}"]) for column in COMPONENT_COLUMNS
    }
    valid_cut = best_top_rate_for_weights(X, y_valid, best_weights)
    train_cut = best_top_rate_for_weights(X, y_train, best_weights)

    best["recommended_top_rate"] = valid_cut["top_rate"]
    best["recommended_threshold"] = valid_cut["threshold"]
    best["recommended_valid_f1"] = valid_cut["f1"]
    best["recommended_valid_precision"] = valid_cut["precision"]
    best["recommended_valid_recall"] = valid_cut["recall"]
    best["cut_selected_on"] = "validation"
    # 참고용: train에서 고른 컷과 비교하면 컷의 안정성을 가늠할 수 있다.
    best["train_optimal_top_rate"] = train_cut["top_rate"]
    best["train_optimal_f1"] = train_cut["f1"]

    save_json(best, output_dir / "recommended_weight_threshold.json")

    print("\n가중치 튜닝 완료")
    print(f"결과 폴더: {output_dir}")
    print("\n로지스틱 회귀 기반 선형 진단")
    print(json.dumps(logistic_diagnostic, ensure_ascii=False, indent=2))
    for key, value in logistic_weights.items():
        print(f"- {key}: {value:.4f}")
    print("\n탐색 기반 추천 후보")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    print("\n0/1 컷(top_rate) 안정성 점검")
    print(f"- train 최적 컷: {train_cut['top_rate']:.4f} (f1={train_cut['f1']:.4f})")
    print(f"- valid 최적 컷: {valid_cut['top_rate']:.4f} (f1={valid_cut['f1']:.4f})")
    print("  → 두 컷이 비슷하면 안정적, 크게 다르면 과적합 의심")
    print(f"- 실제 적용 컷(추천): {valid_cut['top_rate']:.4f} (검증연도 기준)")


if __name__ == "__main__":
    main()
