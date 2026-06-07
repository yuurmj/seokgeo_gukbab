import pandas as pd
from pathlib import Path

# 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "weather_features_by_pole.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# 결과 폴더 생성
OUTPUT_DIR.mkdir(exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(DATA_PATH)

# 데이터 확인
print(df.head())
print(df.columns)
print(df.shape)


#필수 컬럼들 확인
required_cols = [
    "pole_id",
    "pole_lon",
    "pole_lat",
    "nearest_station_id",
    "nearest_station_name",
    "avg_temp",
    "max_temp",
    "avg_humidity",
    "min_humidity",
    "avg_effective_humidity",
    "min_effective_humidity",
    "total_precipitation",
    "rain_7days_min",
    "max_consecutive_dry_hours",
    "avg_wind_speed",
    "max_wind_speed",
    "strong_wind_hours",
    "max_vapor_pressure_deficit",
    "max_dew_point_depression",
    "yanggan_wind_risk",
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"누락된 컬럼이 있습니다: {missing_cols}")


# 숫자형으로 계산해야 하는 컬럼 목록
numeric_cols = [
    "avg_temp",
    "max_temp",
    "avg_humidity",
    "min_humidity",
    "avg_effective_humidity",
    "min_effective_humidity",
    "total_precipitation",
    "rain_7days_min",
    "max_consecutive_dry_hours",
    "avg_wind_speed",
    "max_wind_speed",
    "strong_wind_hours",
    "max_vapor_pressure_deficit",
    "max_dew_point_depression",
]

# 숫자형 컬럼들을 실제 숫자로 변환
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 숫자형 컬럼의 결측치를 각 컬럼의 중앙값으로 채움
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 양간지풍 위험 여부는 0/1 값이므로 결측치는 0으로 처리
df["yanggan_wind_risk"] = pd.to_numeric(df["yanggan_wind_risk"], errors="coerce").fillna(0)



# 점수 가중치 설정
TEMP_SCORE = 5                  # 기온: 보조 위험 요인
HUMIDITY_SCORE = 10             # 습도: 건조 위험과 직접 관련
EFFECTIVE_HUMIDITY_SCORE = 15   # 실효습도: 산불 위험 판단에 중요한 누적 건조도 지표
RAIN_SCORE = 10                 # 강수 부족: 비가 적을수록 건조 위험 증가
WIND_SCORE = 10                 # 바람: 화재 확산 위험 반영
STRONG_WIND_SCORE = 5           # 강풍 지속시간: 강풍 빈도 보조 반영
DRYNESS_SCORE = 10              # 대기 건조도: 증기압·이슬점 결핍량 반영
YANGGAN_SCORE = 10              # 양간지풍: 강원 영동 지역 산불 확산 위험 반영


# 기준값 설정
AVG_TEMP_THRESHOLD = 25
MAX_TEMP_THRESHOLD = 30
AVG_HUMIDITY_THRESHOLD = 45
MIN_HUMIDITY_THRESHOLD = 30
AVG_EFFECTIVE_HUMIDITY_THRESHOLD = 50
MIN_EFFECTIVE_HUMIDITY_THRESHOLD = 35
TOTAL_PRECIPITATION_THRESHOLD = 20
RAIN_7DAYS_MIN_THRESHOLD = 5
DRY_HOURS_THRESHOLD = 72
AVG_WIND_THRESHOLD = 4
MAX_WIND_THRESHOLD = 8
STRONG_WIND_HOURS_THRESHOLD = 24
MAX_VPD_THRESHOLD = 20
MAX_DEW_POINT_DEPRESSION_THRESHOLD = 15
DECISION_THRESHOLD = 60


#점수 계산
# 위험 점수 초기화
df["temp_score"] = 0
df["humidity_score"] = 0
df["effective_humidity_score"] = 0
df["rain_score"] = 0
df["wind_score"] = 0
df["dryness_score"] = 0
df["yanggan_score"] = 0


# 평균 기온이 기준 이상이면 고온으로 인한 건조 위험 점수 추가
df.loc[df["avg_temp"] >= AVG_TEMP_THRESHOLD, "temp_score"] += TEMP_SCORE

# 최고 기온이 기준 이상이면 극단적 고온 위험 점수 추가
df.loc[df["max_temp"] >= MAX_TEMP_THRESHOLD, "temp_score"] += TEMP_SCORE


# 평균 습도가 기준 이하이면 전체적으로 건조한 상태로 보고 위험 점수 추가
df.loc[df["avg_humidity"] <= AVG_HUMIDITY_THRESHOLD, "humidity_score"] += HUMIDITY_SCORE

# 최저 습도가 기준 이하이면 가장 건조했던 순간의 위험 점수 추가
df.loc[df["min_humidity"] <= MIN_HUMIDITY_THRESHOLD, "humidity_score"] += HUMIDITY_SCORE


# 평균 실효습도가 기준 이하이면 누적 건조 상태로 보고 위험 점수 추가
df.loc[df["avg_effective_humidity"] <= AVG_EFFECTIVE_HUMIDITY_THRESHOLD, "effective_humidity_score"] += EFFECTIVE_HUMIDITY_SCORE

# 최저 실효습도가 기준 이하이면 극단적인 산불 위험 건조 상태로 보고 점수 추가
df.loc[df["min_effective_humidity"] <= MIN_EFFECTIVE_HUMIDITY_THRESHOLD, "effective_humidity_score"] += EFFECTIVE_HUMIDITY_SCORE


# 전체 누적 강수량이 기준 이하이면 강수 부족으로 인한 건조 위험 점수 추가
df.loc[df["total_precipitation"] <= TOTAL_PRECIPITATION_THRESHOLD, "rain_score"] += RAIN_SCORE

# 가장 건조했던 7일 동안의 강수량이 기준 이하이면 단기 건조 위험 점수 추가
df.loc[df["rain_7days_min"] <= RAIN_7DAYS_MIN_THRESHOLD, "rain_score"] += RAIN_SCORE

# 비가 오지 않은 시간이 기준 이상으로 길면 장기 건조 위험 점수 추가
df.loc[df["max_consecutive_dry_hours"] >= DRY_HOURS_THRESHOLD, "rain_score"] += RAIN_SCORE


# 평균 풍속이 기준 이상이면 지속적인 바람에 의한 확산 위험 점수 추가
df.loc[df["avg_wind_speed"] >= AVG_WIND_THRESHOLD, "wind_score"] += WIND_SCORE

# 최대 풍속이 기준 이상이면 순간 강풍에 의한 확산 위험 점수 추가
df.loc[df["max_wind_speed"] >= MAX_WIND_THRESHOLD, "wind_score"] += WIND_SCORE

# 강풍 시간이 기준 이상이면 강풍 발생 빈도에 따른 위험 점수 추가
df.loc[df["strong_wind_hours"] >= STRONG_WIND_HOURS_THRESHOLD, "wind_score"] += STRONG_WIND_SCORE


# 최대 증기압 결핍량이 기준 이상이면 대기가 매우 건조한 상태로 보고 위험 점수 추가
df.loc[df["max_vapor_pressure_deficit"] >= MAX_VPD_THRESHOLD, "dryness_score"] += DRYNESS_SCORE

# 최대 이슬점 결핍량이 기준 이상이면 공기 건조도가 높다고 보고 위험 점수 추가
df.loc[df["max_dew_point_depression"] >= MAX_DEW_POINT_DEPRESSION_THRESHOLD, "dryness_score"] += DRYNESS_SCORE


# 양간지풍 위험 조건에 해당하면 강원 영동 산불 확산 위험 점수 추가
df.loc[df["yanggan_wind_risk"] == 1, "yanggan_score"] += YANGGAN_SCORE

# 뭐가 주요 원인인지 추정하기 위함
# 8. 총 위험 점수 계산
score_cols = [
    "temp_score",
    "humidity_score",
    "effective_humidity_score",
    "rain_score",
    "wind_score",
    "dryness_score",
    "yanggan_score",
]

df["rule_risk_score"] = df[score_cols].sum(axis=1)

# # 점수 최대값 제한
# df["rule_risk_score"] = df["rule_risk_score"].clip(upper=100)



# 0/1 판별
df["rule_decision"] = 0
df.loc[df["rule_risk_score"] >= DECISION_THRESHOLD, "rule_decision"] = 1


# 등급 나누기
def assign_risk_level(score):
    if score >= 80:
        return "1등급 매우 위험"
    elif score >= 60:
        return "2등급 위험"
    elif score >= 40:
        return "3등급 주의"
    else:
        return "4등급 낮음"

df["rule_risk_level"] = df["rule_risk_score"].apply(assign_risk_level)


# 등급 안 우선순위 생성
df["rule_priority_rank"] = df.groupby("rule_risk_level")["rule_risk_score"].rank(
    method="first",
    ascending=False
).astype(int)


#결과 저장
result_path = OUTPUT_DIR / "rule_based_result.csv"
submission_path = OUTPUT_DIR / "submission.csv"

df.to_csv(result_path, index=False, encoding="utf-8-sig")

submission = df[["pole_id", "pole_lon", "pole_lat", "rule_decision"]].copy()
submission = submission.rename(columns={
    "pole_lon": "lon",
    "pole_lat": "lat",
    "rule_decision": "decision"
})

submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

print("규칙기반모델 결과 생성 완료")
print(df[["pole_id", "rule_risk_score", "rule_decision", "rule_risk_level", "rule_priority_rank"]].head())



# 확인용
print("\n위험등급 분포")
print(df["rule_risk_level"].value_counts())

print("\ndecision 분포")
print(df["rule_decision"].value_counts())

print("\n상위 위험 전신주 10개")
print(
    df[
        [
            "pole_id",
            "rule_risk_score",
            "rule_decision",
            "rule_risk_level",
            "rule_priority_rank",
            "temp_score",
            "humidity_score",
            "rain_score",
            "wind_score",
            "dryness_score",
            "yanggan_score",
        ]
    ].sort_values("rule_risk_score", ascending=False).head(10)
)