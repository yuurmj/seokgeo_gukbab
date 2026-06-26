# 라이브러리 불러오기 및 한글 출력 설정
import json
import pandas as pd
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")


# 테스트 모드 설정
TEST_MODE = False
NROWS = 10000
nrows = NROWS if TEST_MODE else None


# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHT_TUNING_PATH = OUTPUT_DIR / "weight_tuning" / "recommended_weight_threshold.json"


# 0~1 정규화 함수
def minmax(series):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(0, index=series.index)

    return (series - min_value) / (max_value - min_value)


# 값이 클수록 위험한 변수
def risk_high(series):
    return minmax(series)


# 값이 작을수록 위험한 변수
def risk_low(series):
    return 1 - minmax(series)


# 기상위험도 상위 가중치
WEATHER_DRYNESS_WEIGHT = 0.55
WEATHER_WIND_WEIGHT = 0.30
WEATHER_TEMPERATURE_WEIGHT = 0.15


# 건조위험도 내부 가중치
DRY_AVG_HUMIDITY_WEIGHT = 0.15
DRY_MIN_HUMIDITY_WEIGHT = 0.15
DRY_AVG_EFFECTIVE_HUMIDITY_WEIGHT = 0.20
DRY_MIN_EFFECTIVE_HUMIDITY_WEIGHT = 0.20
DRY_TOTAL_PRECIPITATION_WEIGHT = 0.10
DRY_CONSECUTIVE_DRY_HOURS_WEIGHT = 0.10
DRY_VAPOR_PRESSURE_DEFICIT_WEIGHT = 0.10


# 바람위험도 내부 가중치
WIND_AVG_WIND_SPEED_WEIGHT = 0.25
WIND_MAX_WIND_SPEED_WEIGHT = 0.30
WIND_STRONG_WIND_HOURS_WEIGHT = 0.25
WIND_YANGGAN_WEIGHT = 0.20


# 고온위험도 내부 가중치
TEMP_AVG_TEMP_WEIGHT = 0.40
TEMP_MAX_TEMP_WEIGHT = 0.60


# 공간위험도 가중치
SPATIAL_RATIO_100M_WEIGHT = 0.18
SPATIAL_RATIO_300M_WEIGHT = 0.24
SPATIAL_RATIO_500M_WEIGHT = 0.06
SPATIAL_DISTANCE_WEIGHT = 0.15
SPATIAL_SCALED_WEIGHT = 0.15
SPATIAL_SLOPE_WEIGHT = 0.14
SPATIAL_ELEVATION_WEIGHT = 0.08


# 최종위험도 가중치
FINAL_WEATHER_WEIGHT = 0.50
FINAL_SPATIAL_WEIGHT = 0.30
FINAL_INTERACTION_WEIGHT = 0.20
FINAL_TOP_RATE = None


# 가중치 튜닝 결과 적용
if WEIGHT_TUNING_PATH.exists():
    with WEIGHT_TUNING_PATH.open(encoding="utf-8") as file:
        tuned = json.load(file)

    FINAL_WEATHER_WEIGHT = tuned.get(
        "weight_weather_risk_score",
        FINAL_WEATHER_WEIGHT,
    )
    FINAL_SPATIAL_WEIGHT = tuned.get(
        "weight_spatial_risk_score",
        FINAL_SPATIAL_WEIGHT,
    )
    FINAL_INTERACTION_WEIGHT = tuned.get(
        "weight_interaction_risk_score",
        FINAL_INTERACTION_WEIGHT,
    )
    FINAL_TOP_RATE = tuned.get("train_top_rate", tuned.get("top_rate"))

    print("\n가중치 튜닝 결과 적용")
    print(f"- weather: {FINAL_WEATHER_WEIGHT:.4f}")
    print(f"- spatial: {FINAL_SPATIAL_WEIGHT:.4f}")
    print(f"- interaction: {FINAL_INTERACTION_WEIGHT:.4f}")
    if FINAL_TOP_RATE is not None:
        print(f"- decision top_rate: {FINAL_TOP_RATE:.4f}")
else:
    print("\n가중치 튜닝 결과 없음: 기본 최종위험도 가중치 사용")


# 사용할 기상 데이터 선택
WEATHER_MODE = "train" 

if WEATHER_MODE == "train":
    STATION_WEATHER_PATH = DATA_DIR / "weather_train_station_average_202011_202305_reduced.csv"
elif WEATHER_MODE == "validation":
    STATION_WEATHER_PATH = DATA_DIR / "weather_validation_station_202311_202405_reduced.csv"
else:
    raise ValueError("WEATHER_MODE는 'train' 또는 'validation'만 가능합니다.")


# 출력 파일 경로 설정
OUTPUT_PATH = OUTPUT_DIR / f"station_weather_risk_score_{WEATHER_MODE}.csv"
POLE_WEATHER_OUTPUT_PATH = OUTPUT_DIR / f"pole_weather_risk_score_{WEATHER_MODE}.csv"
SPATIAL_OUTPUT_PATH = OUTPUT_DIR / f"pole_spatial_risk_score_{WEATHER_MODE}.csv"
FINAL_OUTPUT_PATH = OUTPUT_DIR / f"final_risk_result_{WEATHER_MODE}.csv"
SUBMISSION_OUTPUT_PATH = OUTPUT_DIR / f"submission_{WEATHER_MODE}.csv"


# 관측소 기상 데이터 읽기
station_df = pd.read_csv(STATION_WEATHER_PATH, encoding="utf-8-sig")

print("관측소 기상 데이터 확인")
print(station_df.head())
print(station_df.columns)
print(station_df.shape)


# 관측소 기상 데이터 필수 컬럼 확인
required_cols = [
    "station_id",
    "station_name",
    "region_type",
    "year_count",
    "avg_temp",
    "max_temp",
    "avg_humidity",
    "min_humidity",
    "avg_effective_humidity",
    "min_effective_humidity",
    "total_precipitation",
    "max_consecutive_dry_hours",
    "avg_wind_speed",
    "max_wind_speed",
    "strong_wind_hours",
    "max_vapor_pressure_deficit",
    "max_dew_point_depression",  # max_dew_point_depression은 참고 변수로 보유하되, 현재 기상위험도 가중식에는 반영하지 않음
    "yanggan_wind_risk",
]

missing_cols = [col for col in required_cols if col not in station_df.columns]

if missing_cols:
    raise ValueError(f"누락된 컬럼이 있습니다: {missing_cols}")


# 기상 변수 숫자형 변환 및 결측치 처리
numeric_cols = [
    "year_count",
    "avg_temp",
    "max_temp",
    "avg_humidity",
    "min_humidity",
    "avg_effective_humidity",
    "min_effective_humidity",
    "total_precipitation",
    "max_consecutive_dry_hours",
    "avg_wind_speed",
    "max_wind_speed",
    "strong_wind_hours",
    "max_vapor_pressure_deficit",
    "max_dew_point_depression",
    "yanggan_wind_risk",
]


#숫자 변환, 결측치 처리
for col in numeric_cols:
    station_df[col] = pd.to_numeric(station_df[col], errors="coerce")

station_df[numeric_cols] = station_df[numeric_cols].fillna(
    station_df[numeric_cols].median()
)

station_df["yanggan_wind_risk"] = station_df["yanggan_wind_risk"].fillna(0)


# 기상 변수별 0~1 위험값 생성
station_df["avg_humidity_risk"] = risk_low(station_df["avg_humidity"])
station_df["min_humidity_risk"] = risk_low(station_df["min_humidity"])
station_df["avg_effective_humidity_risk"] = risk_low(station_df["avg_effective_humidity"])
station_df["min_effective_humidity_risk"] = risk_low(station_df["min_effective_humidity"])
station_df["total_precipitation_risk"] = risk_low(station_df["total_precipitation"])
station_df["max_consecutive_dry_hours_risk"] = risk_high(station_df["max_consecutive_dry_hours"])
station_df["max_vapor_pressure_deficit_risk"] = risk_high(station_df["max_vapor_pressure_deficit"])

station_df["avg_wind_speed_risk"] = risk_high(station_df["avg_wind_speed"])
station_df["max_wind_speed_risk"] = risk_high(station_df["max_wind_speed"])
station_df["strong_wind_hours_risk"] = risk_high(station_df["strong_wind_hours"])
station_df["yanggan_wind_risk_value"] = station_df["yanggan_wind_risk"]

station_df["avg_temp_risk"] = risk_high(station_df["avg_temp"])
station_df["max_temp_risk"] = risk_high(station_df["max_temp"])


# 건조위험도 계산
station_df["dryness_risk"] = (
    station_df["avg_humidity_risk"] * DRY_AVG_HUMIDITY_WEIGHT
    + station_df["min_humidity_risk"] * DRY_MIN_HUMIDITY_WEIGHT
    + station_df["avg_effective_humidity_risk"] * DRY_AVG_EFFECTIVE_HUMIDITY_WEIGHT
    + station_df["min_effective_humidity_risk"] * DRY_MIN_EFFECTIVE_HUMIDITY_WEIGHT
    + station_df["total_precipitation_risk"] * DRY_TOTAL_PRECIPITATION_WEIGHT
    + station_df["max_consecutive_dry_hours_risk"] * DRY_CONSECUTIVE_DRY_HOURS_WEIGHT
    + station_df["max_vapor_pressure_deficit_risk"] * DRY_VAPOR_PRESSURE_DEFICIT_WEIGHT
)


# 바람위험도 계산
station_df["wind_risk"] = (
    station_df["avg_wind_speed_risk"] * WIND_AVG_WIND_SPEED_WEIGHT
    + station_df["max_wind_speed_risk"] * WIND_MAX_WIND_SPEED_WEIGHT
    + station_df["strong_wind_hours_risk"] * WIND_STRONG_WIND_HOURS_WEIGHT
    + station_df["yanggan_wind_risk_value"] * WIND_YANGGAN_WEIGHT
)


# 고온위험도 계산
station_df["temperature_risk"] = (
    station_df["avg_temp_risk"] * TEMP_AVG_TEMP_WEIGHT
    + station_df["max_temp_risk"] * TEMP_MAX_TEMP_WEIGHT
)


# 관측소별 최종 기상위험도 계산
station_df["station_weather_risk_score"] = (
    station_df["dryness_risk"] * WEATHER_DRYNESS_WEIGHT
    + station_df["wind_risk"] * WEATHER_WIND_WEIGHT
    + station_df["temperature_risk"] * WEATHER_TEMPERATURE_WEIGHT
)


# 관측소별 기상 위험 점수 결과 저장
station_weather_result = station_df[
    [
        "station_id",
        "station_name",
        "region_type",
        "station_weather_risk_score",
        "dryness_risk",
        "wind_risk",
        "temperature_risk",
        "avg_humidity_risk",
        "min_humidity_risk",
        "avg_effective_humidity_risk",
        "min_effective_humidity_risk",
        "total_precipitation_risk",
        "max_consecutive_dry_hours_risk",
        "max_vapor_pressure_deficit_risk",
        "avg_wind_speed_risk",
        "max_wind_speed_risk",
        "strong_wind_hours_risk",
        "yanggan_wind_risk_value",
        "avg_temp_risk",
        "max_temp_risk",
    ]
].copy()

station_weather_result.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# 관측소별 기상 위험 점수 결과 확인
print("\n관측소별 기상 위험 점수 생성 완료")
print(station_weather_result.head())

print("\nstation_weather_risk_score 분포")
print(station_weather_result["station_weather_risk_score"].describe())

print("\n관측소 위험 점수 순위")

station_rank = station_weather_result.sort_values(
    "station_weather_risk_score",
    ascending=False
).reset_index(drop=True)

station_rank["rank"] = station_rank.index + 1

print(
    station_rank[
        [
            "rank",
            "station_weather_risk_score",
            "station_name"
        ]
    ]
)


# 전신주별 가까운 관측소 3개 및 가중치 데이터 읽기
POLE_STATION_PATH = DATA_DIR / "pole_stations3_add_w.csv"

pole_df = pd.read_csv(POLE_STATION_PATH, nrows=nrows)

print("\n전신주-관측소 가중치 데이터 확인")
print(pole_df.head())
print(pole_df.shape)


# 전신주-관측소 가중치 데이터 필수 컬럼 확인
required_pole_station_cols = [
    "pole_id",
    "lon",
    "lat",
    "station_id_1",
    "station_id_2",
    "station_id_3",
    "w1",
    "w2",
    "w3",
]

missing_pole_station_cols = [
    col for col in required_pole_station_cols
    if col not in pole_df.columns
]

if missing_pole_station_cols:
    raise ValueError(f"전신주-관측소 파일에 누락된 컬럼이 있습니다: {missing_pole_station_cols}")


# 관측소별 기상 위험 점수를 딕셔너리로 변환
station_score_dict = dict(
    zip(
        station_weather_result["station_id"],
        station_weather_result["station_weather_risk_score"]
    )
)


# 전신주별 가까운 관측소 3개의 기상 위험 점수 매핑
pole_df["station_score_1"] = pole_df["station_id_1"].map(station_score_dict)
pole_df["station_score_2"] = pole_df["station_id_2"].map(station_score_dict)
pole_df["station_score_3"] = pole_df["station_id_3"].map(station_score_dict)


# 관측소 점수 매핑 누락 여부 확인
station_score_cols = ["station_score_1", "station_score_2", "station_score_3"]

missing_station_score_count = pole_df[station_score_cols].isna().sum().sum()

if missing_station_score_count > 0:
    missing_rows = pole_df[pole_df[station_score_cols].isna().any(axis=1)]
    print(
        missing_rows[
            [
                "pole_id",
                "station_id_1",
                "station_id_2",
                "station_id_3"
            ]
        ].head(20)
    )
    raise ValueError("관측소 점수가 매핑되지 않은 전신주가 있습니다.")


# 전신주별 기상 위험 점수 계산
pole_df["weather_risk_score"] = (
    pole_df["station_score_1"] * pole_df["w1"]
    + pole_df["station_score_2"] * pole_df["w2"]
    + pole_df["station_score_3"] * pole_df["w3"]
)


# 전신주별 기상 위험 점수 결과 저장
pole_weather_result = pole_df[
    [
        "pole_id",
        "lon",
        "lat",
        "weather_risk_score"
    ]
].copy()

pole_weather_result.to_csv(
    POLE_WEATHER_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# 전신주별 기상 위험 점수 결과 확인
print("\n전신주별 weather_risk_score 생성 완료")
print(pole_weather_result.head())

print("\nweather_risk_score 통계")
print(pole_weather_result["weather_risk_score"].describe())

weather_rank = pole_weather_result.sort_values(
    "weather_risk_score",
    ascending=False
).reset_index(drop=True)

weather_rank["rank"] = weather_rank.index + 1

print("\n전신주 weather_risk_score 상위 20개")
print(
    weather_rank[
        [
            "rank",
            "pole_id",
            "weather_risk_score"
        ]
    ].head(20)
)

# step 3
# 공간 변수 파일 경로 설정

# 산림 변수 파일
FOREST_DISTANCE_PATH = DATA_DIR / "pole_distance_selected.csv"
FOREST_AREA_PATH = DATA_DIR / "gangwon_pole_scaled_risk.csv"
FOREST_RATIO_100_PATH = DATA_DIR / "pole_forest_ratio_100m.csv"
FOREST_RATIO_300_PATH = DATA_DIR / "pole_forest_ratio_300m.csv"
FOREST_RATIO_500_PATH = DATA_DIR / "pole_forest_ratio_500m.csv"

# 지형 변수 파일
SLOPE_PATH = DATA_DIR / "slope_mean.csv"
HEIGHT_PATH = DATA_DIR / "pole_locations_with_height.csv"


# 전신주별 weather_risk_score 결과 읽기
weather_df = pd.read_csv(POLE_WEATHER_OUTPUT_PATH, nrows=nrows)


# 공간 변수 데이터 읽기
distance_df = pd.read_csv(FOREST_DISTANCE_PATH, nrows=nrows)
area_df = pd.read_csv(FOREST_AREA_PATH, nrows=nrows)
ratio100_df = pd.read_csv(FOREST_RATIO_100_PATH, nrows=nrows)
ratio300_df = pd.read_csv(FOREST_RATIO_300_PATH, nrows=nrows)
ratio500_df = pd.read_csv(FOREST_RATIO_500_PATH, nrows=nrows)
slope_df = pd.read_csv(SLOPE_PATH, nrows=nrows)
height_df = pd.read_csv(HEIGHT_PATH, nrows=nrows)


# 산림거리 변수 정리
distance_df = distance_df[["pole_id", "distance"]].rename(
    columns={"distance": "forest_distance"}
)


# 산림 전체 크기 변수 정리: f_area_sum은 사용하지 않고 spatial_risk_scaled만 사용
area_df = area_df[["pole_id", "spatial_risk_scaled"]]


# 반경별 산림비율 변수 정리
ratio100_df = ratio100_df[["pole_id", "forest_ratio_100m"]]
ratio300_df = ratio300_df[["pole_id", "forest_ratio_300m"]]
ratio500_df = ratio500_df[["pole_id", "forest_ratio_500m"]]


# 경사도 변수 정리
slope_df = slope_df[["pole_id", "slopemean"]].rename(
    columns={"slopemean": "pole_slope"}
)


# 고도 변수 정리
height_df = height_df[["pole_id", "height1"]].rename(
    columns={"height1": "pole_height"}
)


# pole_id 기준으로 기상 점수와 공간 변수 병합
spatial_df = weather_df.merge(distance_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(area_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(ratio100_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(ratio300_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(ratio500_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(slope_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(height_df, on="pole_id", how="left")


# 공간 변수 누락 여부 확인
spatial_cols = [
    "forest_distance",
    "spatial_risk_scaled",
    "forest_ratio_100m",
    "forest_ratio_300m",
    "forest_ratio_500m",
    "pole_slope",
    "pole_height",
]

missing_spatial_count = spatial_df[spatial_cols].isna().sum().sum()

if missing_spatial_count > 0:
    missing_rows = spatial_df[spatial_df[spatial_cols].isna().any(axis=1)]
    print("\n공간 변수가 누락된 전신주 예시")
    print(missing_rows[["pole_id"] + spatial_cols].head(20))
    raise ValueError("공간 변수가 매핑되지 않은 전신주가 있습니다.")


# 공간 변수 숫자형 변환
for col in spatial_cols:
    spatial_df[col] = pd.to_numeric(spatial_df[col], errors="coerce")


# 숫자형 변환 이후 결측치 재확인
missing_after_numeric_count = spatial_df[spatial_cols].isna().sum().sum()

if missing_after_numeric_count > 0:
    missing_rows = spatial_df[spatial_df[spatial_cols].isna().any(axis=1)]
    print("\n숫자형 변환 후 결측치가 생긴 전신주 예시")
    print(missing_rows[["pole_id"] + spatial_cols].head(20))
    raise ValueError("공간 변수 숫자형 변환 과정에서 결측치가 발생했습니다.")


# 공간 변수별 0~1 위험값 생성
spatial_df["forest_ratio_100m_risk"] = risk_high(spatial_df["forest_ratio_100m"])
spatial_df["forest_ratio_300m_risk"] = risk_high(spatial_df["forest_ratio_300m"])
spatial_df["forest_ratio_500m_risk"] = risk_high(spatial_df["forest_ratio_500m"])

spatial_df["forest_distance_risk"] = risk_low(spatial_df["forest_distance"])

# spatial_risk_scaled는 이미 0~1 위험값이므로 그대로 사용
spatial_df["spatial_risk_scaled_risk"] = spatial_df["spatial_risk_scaled"]

spatial_df["slope_risk"] = risk_high(spatial_df["pole_slope"])
spatial_df["elevation_risk"] = risk_high(spatial_df["pole_height"])


# 공간위험도 계산
spatial_df["spatial_risk_score"] = (
    spatial_df["forest_ratio_100m_risk"] * SPATIAL_RATIO_100M_WEIGHT
    + spatial_df["forest_ratio_300m_risk"] * SPATIAL_RATIO_300M_WEIGHT
    + spatial_df["forest_ratio_500m_risk"] * SPATIAL_RATIO_500M_WEIGHT
    + spatial_df["forest_distance_risk"] * SPATIAL_DISTANCE_WEIGHT
    + spatial_df["spatial_risk_scaled_risk"] * SPATIAL_SCALED_WEIGHT
    + spatial_df["slope_risk"] * SPATIAL_SLOPE_WEIGHT
    + spatial_df["elevation_risk"] * SPATIAL_ELEVATION_WEIGHT
)


# 공간 위험 점수 결과 저장
spatial_result = spatial_df[
    [
        "pole_id",
        "lon",
        "lat",
        "weather_risk_score",
        "forest_distance",
        "spatial_risk_scaled",
        "forest_ratio_100m",
        "forest_ratio_300m",
        "forest_ratio_500m",
        "pole_slope",
        "pole_height",
        "forest_ratio_100m_risk",
        "forest_ratio_300m_risk",
        "forest_ratio_500m_risk",
        "forest_distance_risk",
        "spatial_risk_scaled_risk",
        "slope_risk",
        "elevation_risk",
        "spatial_risk_score",
    ]
].copy()

spatial_result.to_csv(
    SPATIAL_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# 공간 위험 점수 결과 확인
print("\n전신주별 spatial_risk_score 생성 완료")
print(spatial_result.head())

print("\nspatial_risk_score 통계")
print(spatial_result["spatial_risk_score"].describe())

print("\n공간 위험 점수 상위 20개")
spatial_rank = spatial_result.sort_values(
    "spatial_risk_score",
    ascending=False
).reset_index(drop=True)

spatial_rank["rank"] = spatial_rank.index + 1

print(
    spatial_rank[
        [
            "rank",
            "pole_id",
            "spatial_risk_score",
            "forest_ratio_100m_risk",
            "forest_ratio_300m_risk",
            "forest_ratio_500m_risk",
            "forest_distance_risk",
            "spatial_risk_scaled_risk",
            "slope_risk",
            "elevation_risk",
        ]
    ].head(20)
)

# 최종 위험도 계산용 데이터 읽기
final_df = pd.read_csv(SPATIAL_OUTPUT_PATH, nrows=nrows)

print("\n최종 위험도 계산 시작")
print(final_df.shape)


# 최종 위험도
# 상호작용위험도 계산
final_df["interaction_risk_score"] = (
    final_df["weather_risk_score"]
    * final_df["spatial_risk_score"]
)


# 최종위험도 계산
final_df["final_risk_score"] = (
    final_df["weather_risk_score"] * FINAL_WEATHER_WEIGHT
    + final_df["spatial_risk_score"] * FINAL_SPATIAL_WEIGHT
    + final_df["interaction_risk_score"] * FINAL_INTERACTION_WEIGHT
)


# 위험 여부 판단 기준 설정
if FINAL_TOP_RATE is None:
    decision_threshold = final_df["final_risk_score"].median()
else:
    decision_threshold = final_df["final_risk_score"].quantile(1 - FINAL_TOP_RATE)


# 최종 위험 여부 생성
final_df["decision"] = (
    final_df["final_risk_score"] >= decision_threshold
).astype(int)


# 위험도 4등급 생성
# 1(매우높음) = decision==1 (점검 대상으로 찍은 전신주)
# 나머지(decision==0)는 final_risk_score 기준으로 3등분 → 2(높음)/3(보통)/4(낮음)
final_df["risk_level"] = "1(매우높음)"

decision_zero_mask = final_df["decision"] == 0
final_df.loc[decision_zero_mask, "risk_level"] = pd.qcut(
    final_df.loc[decision_zero_mask, "final_risk_score"].rank(method="first"),
    q=3,
    labels=[
        "4(낮음)",
        "3(보통)",
        "2(높음)"
    ]
).astype(str)

# 등급을 1→4 순서가 유지되는 범주형으로 정리
final_df["risk_level"] = pd.Categorical(
    final_df["risk_level"],
    categories=["1(매우높음)", "2(높음)", "3(보통)", "4(낮음)"],
    ordered=True
)


# 등급 안 우선순위 생성
final_df["priority_rank"] = final_df.groupby("risk_level")["final_risk_score"].rank(
    method="first",
    ascending=False
).astype(int)


# 분석용 최종 결과 저장
final_result = final_df[
    [
        "pole_id",
        "lon",
        "lat",
        "weather_risk_score",
        "spatial_risk_score",
        "interaction_risk_score",
        "final_risk_score",
        "decision",
        "risk_level",
        "priority_rank"
    ]
].copy()

final_result.to_csv(
    FINAL_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# 제출용 파일 저장
submission = final_df[
    [
        "pole_id",
        "lon",
        "lat",
        "decision"
    ]
].copy()

submission.to_csv(
    SUBMISSION_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# 최종 결과 확인
print("\n최종 위험도 생성 완료")

print("\ndecision 분포")
print(final_result["decision"].value_counts())

print("\nrisk_level 분포")
print(final_result["risk_level"].value_counts())

print("\n제출용 submission.csv 생성 완료")
print(submission.head())


# 최종 위험도 순위 확인
final_rank = final_result.sort_values(
    "final_risk_score",
    ascending=False
).reset_index(drop=True)

final_rank["rank"] = final_rank.index + 1

print("\n최종 위험도 상위 20개")
print(
    final_rank[
        [
            "rank",
            "pole_id",
            "final_risk_score",
            "risk_level"
        ]
    ].head(20)
)