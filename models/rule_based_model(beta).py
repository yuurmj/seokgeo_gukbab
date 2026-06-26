import pandas as pd
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")


# 테스트 모드(만 개 제한)
TEST_MODE = True
NROWS = 10000
nrows = NROWS if TEST_MODE else None


# step 1
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_WEATHER_PATH = DATA_DIR / "weather_featured_station_5year_average.csv"
OUTPUT_PATH = OUTPUT_DIR / "station_weather_risk_score.csv"

station_df = pd.read_csv(STATION_WEATHER_PATH, encoding="utf-8-sig")

print("관측소 기상 데이터 확인")
print(station_df.head())
print(station_df.columns)
print(station_df.shape)

# 필수 컬럼 확인
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
    "max_dew_point_depression",
    "yanggan_wind_risk",
]

missing_cols = [col for col in required_cols if col not in station_df.columns]

if missing_cols:
    raise ValueError(f"누락된 컬럼이 있습니다: {missing_cols}")


# 숫자형 변환 및 결측치 처리
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

for col in numeric_cols:
    station_df[col] = pd.to_numeric(station_df[col], errors="coerce")

station_df[numeric_cols] = station_df[numeric_cols].fillna(
    station_df[numeric_cols].median()
)

station_df["yanggan_wind_risk"] = station_df["yanggan_wind_risk"].fillna(0)



# 관측소별 기온 점수 계산

# 점수 초기화
station_df["station_temp_score"] = 0
station_df["station_humidity_score"] = 0
station_df["station_effective_humidity_score"] = 0
station_df["station_rain_score"] = 0
station_df["station_wind_score"] = 0
station_df["station_dryness_score"] = 0
station_df["station_yanggan_score"] = 0



# 기온 점수
station_df.loc[station_df["avg_temp"] >= 20, "station_temp_score"] += 2
station_df.loc[station_df["avg_temp"] >= 25, "station_temp_score"] += 3

station_df.loc[station_df["max_temp"] >= 30, "station_temp_score"] += 3
station_df.loc[station_df["max_temp"] >= 35, "station_temp_score"] += 4



# 습도 점수
station_df.loc[station_df["avg_humidity"] <= 50, "station_humidity_score"] += 2
station_df.loc[station_df["avg_humidity"] <= 40, "station_humidity_score"] += 3

station_df.loc[station_df["min_humidity"] <= 30, "station_humidity_score"] += 3
station_df.loc[station_df["min_humidity"] <= 20, "station_humidity_score"] += 4



# 실효습도 점수
station_df.loc[
    station_df["avg_effective_humidity"] <= 60,
    "station_effective_humidity_score"
] += 3

station_df.loc[
    station_df["avg_effective_humidity"] <= 50,
    "station_effective_humidity_score"
] += 4

station_df.loc[
    station_df["min_effective_humidity"] <= 35,
    "station_effective_humidity_score"
] += 4

station_df.loc[
    station_df["min_effective_humidity"] <= 25,
    "station_effective_humidity_score"
] += 5



# 강수 점수
station_df.loc[
    station_df["total_precipitation"] <= 1200,
    "station_rain_score"
] += 3

station_df.loc[
    station_df["total_precipitation"] <= 1000,
    "station_rain_score"
] += 3

station_df.loc[
    station_df["max_consecutive_dry_hours"] >= 500,
    "station_rain_score"
] += 2

station_df.loc[
    station_df["max_consecutive_dry_hours"] >= 700,
    "station_rain_score"
] += 3



# 바람 점수
station_df.loc[station_df["avg_wind_speed"] >= 2, "station_wind_score"] += 2
station_df.loc[station_df["avg_wind_speed"] >= 3, "station_wind_score"] += 3

station_df.loc[station_df["max_wind_speed"] >= 10, "station_wind_score"] += 4
station_df.loc[station_df["max_wind_speed"] >= 15, "station_wind_score"] += 5

station_df.loc[station_df["strong_wind_hours"] >= 500, "station_wind_score"] += 2
station_df.loc[station_df["strong_wind_hours"] >= 800, "station_wind_score"] += 3



# 대기 건조도 점수
station_df.loc[
    station_df["max_vapor_pressure_deficit"] >= 20,
    "station_dryness_score"
] += 4

station_df.loc[
    station_df["max_vapor_pressure_deficit"] >= 30,
    "station_dryness_score"
] += 4

station_df.loc[
    station_df["max_dew_point_depression"] >= 20,
    "station_dryness_score"
] += 2

station_df.loc[
    station_df["max_dew_point_depression"] >= 30,
    "station_dryness_score"
] += 3



# 양간지풍 점수
station_df.loc[
    station_df["yanggan_wind_risk"] == 1,
    "station_yanggan_score"
] += 5



# station_weather_risk_score 계산
station_score_cols = [
    "station_temp_score",
    "station_humidity_score",
    "station_effective_humidity_score",
    "station_rain_score",
    "station_wind_score",
    "station_dryness_score",
    "station_yanggan_score",
]

station_df["station_weather_risk_score"] = station_df[station_score_cols].sum(axis=1)



# 결과 저장
station_weather_result = station_df[
    [
        "station_id",
        "station_name",
        "region_type",
        "station_weather_risk_score",
        "station_temp_score",
        "station_humidity_score",
        "station_effective_humidity_score",
        "station_rain_score",
        "station_wind_score",
        "station_dryness_score",
        "station_yanggan_score",
    ]
].copy()

station_weather_result.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

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


# step 2
# 전신주별 가까운 관측소 3개 가중치 데이터 읽기
POLE_STATION_PATH = DATA_DIR / "pole_stations3_add_w.csv"
POLE_WEATHER_OUTPUT_PATH = OUTPUT_DIR / "pole_weather_risk_score.csv"

pole_df = pd.read_csv(POLE_STATION_PATH, nrows=nrows)

print("\n전신주-관측소 가중치 데이터 확인")
print(pole_df.head())
print(pole_df.shape)

# 관측소별 기상 위험 점수를 딕셔너리로 변환
station_score_dict = dict(
    zip(
        station_weather_result["station_id"],
        station_weather_result["station_weather_risk_score"]
    )
)

# 전신주별 가까운 관측소 3개의 위험 점수 매핑
pole_df["station_score_1"] = pole_df["station_id_1"].map(station_score_dict)
pole_df["station_score_2"] = pole_df["station_id_2"].map(station_score_dict)
pole_df["station_score_3"] = pole_df["station_id_3"].map(station_score_dict)

# 관측소 점수가 제대로 매핑되었는지 확인
station_score_cols = ["station_score_1", "station_score_2", "station_score_3"]

missing_station_score_count = pole_df[station_score_cols].isna().sum().sum()

if missing_station_score_count > 0:
    missing_rows = pole_df[pole_df[station_score_cols].isna().any(axis=1)]
    print(missing_rows[["pole_id", "station_id_1", "station_id_2", "station_id_3"]].head(20))
    raise ValueError("관측소 점수가 매핑되지 않은 전신주가 있습니다.")

# 전신주별 weather_risk_score 계산
pole_df["weather_risk_score"] = (
    pole_df["station_score_1"] * pole_df["w1"]
    + pole_df["station_score_2"] * pole_df["w2"]
    + pole_df["station_score_3"] * pole_df["w3"]
)

# 필요한 컬럼만 저장
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

# 결과 확인
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
FOREST_DISTANCE_PATH = DATA_DIR / "pole_distance_selected.csv"
FOREST_AREA_PATH = DATA_DIR / "gangwon_pole_scaled_risk.csv"
SLOPE_PATH = DATA_DIR / "slope_mean.csv"
HEIGHT_PATH = DATA_DIR / "pole_locations_with_height.csv"

SPATIAL_OUTPUT_PATH = OUTPUT_DIR / "pole_spatial_risk_score.csv"

# 전신주별 weather_risk_score 결과 읽기
weather_df = pd.read_csv(POLE_WEATHER_OUTPUT_PATH, nrows=nrows)

# 공간 변수 데이터 읽기
distance_df = pd.read_csv(FOREST_DISTANCE_PATH, nrows=nrows)
area_df = pd.read_csv(FOREST_AREA_PATH, nrows=nrows)
slope_df = pd.read_csv(SLOPE_PATH, nrows=nrows)
height_df = pd.read_csv(HEIGHT_PATH, nrows=nrows)

# 필요한 컬럼만 선택하고 컬럼명 정리
distance_df = distance_df[["pole_id", "distance"]].rename(
    columns={"distance": "forest_distance"}
)

area_df = area_df[["pole_id", "f_area_sum"]].rename(
    columns={"f_area_sum": "forest_area"}
)

slope_df = slope_df[["pole_id", "slopemean"]].rename(
    columns={"slopemean": "pole_slope"}
)

height_df = height_df[["pole_id", "height1"]].rename(
    columns={"height1": "pole_height"}
)

# pole_id 기준으로 weather 결과와 공간 변수 병합
spatial_df = weather_df.merge(distance_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(area_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(slope_df, on="pole_id", how="left")
spatial_df = spatial_df.merge(height_df, on="pole_id", how="left")

# 공간 변수 누락 여부 확인
spatial_cols = [
    "forest_distance",
    "forest_area",
    "pole_slope",
    "pole_height",
]

missing_spatial_count = spatial_df[spatial_cols].isna().sum().sum()

if missing_spatial_count > 0:
    missing_rows = spatial_df[spatial_df[spatial_cols].isna().any(axis=1)]
    print(missing_rows[["pole_id"] + spatial_cols].head(20))
    raise ValueError("공간 변수가 매핑되지 않은 전신주가 있습니다.")

# 숫자형 변환
for col in spatial_cols:
    spatial_df[col] = pd.to_numeric(spatial_df[col], errors="coerce")

# 공간 점수 초기화
spatial_df["forest_distance_score"] = 0
spatial_df["forest_area_score"] = 0
spatial_df["pole_slope_score"] = 0
spatial_df["pole_height_score"] = 0

# 산림거리 점수: 산림과 가까울수록 위험
spatial_df.loc[spatial_df["forest_distance"] <= 500, "forest_distance_score"] += 2
spatial_df.loc[spatial_df["forest_distance"] <= 100, "forest_distance_score"] += 3
spatial_df.loc[spatial_df["forest_distance"] <= 50, "forest_distance_score"] += 4
spatial_df.loc[spatial_df["forest_distance"] <= 30, "forest_distance_score"] += 5

# 산림면적 점수: 주변 산림 면적이 클수록 위험
spatial_df.loc[spatial_df["forest_area"] >= 500000, "forest_area_score"] += 2
spatial_df.loc[spatial_df["forest_area"] >= 1000000, "forest_area_score"] += 3
spatial_df.loc[spatial_df["forest_area"] >= 3000000, "forest_area_score"] += 4
spatial_df.loc[spatial_df["forest_area"] >= 5000000, "forest_area_score"] += 5

# 경사도 점수: 경사가 클수록 확산 위험 증가
spatial_df.loc[spatial_df["pole_slope"] >= 10, "pole_slope_score"] += 2
spatial_df.loc[spatial_df["pole_slope"] >= 20, "pole_slope_score"] += 3
spatial_df.loc[spatial_df["pole_slope"] >= 30, "pole_slope_score"] += 4

# 고도 점수: 보조 지형 변수로 낮은 비중 반영
spatial_df.loc[spatial_df["pole_height"] >= 300, "pole_height_score"] += 1
spatial_df.loc[spatial_df["pole_height"] >= 600, "pole_height_score"] += 2
spatial_df.loc[spatial_df["pole_height"] >= 900, "pole_height_score"] += 2

# spatial_risk_score 계산
spatial_score_cols = [
    "forest_distance_score",
    "forest_area_score",
    "pole_slope_score",
    "pole_height_score",
]

spatial_df["spatial_risk_score"] = spatial_df[spatial_score_cols].sum(axis=1)

# 필요한 컬럼만 저장
spatial_result = spatial_df[
    [
        "pole_id",
        "lon",
        "lat",
        "weather_risk_score",
        "forest_distance",
        "forest_area",
        "pole_slope",
        "pole_height",
        "spatial_risk_score",
    ]
].copy()

spatial_result.to_csv(
    SPATIAL_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

# 결과 확인
print("\n전신주별 spatial_risk_score 생성 완료")
print(spatial_result.head())

print("\nspatial_risk_score 통계")
print(spatial_result["spatial_risk_score"].describe())


# step 4
# 최종 결과 파일 경로 설정
FINAL_OUTPUT_PATH = OUTPUT_DIR / "final_risk_result.csv"
SUBMISSION_OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

# 최종 위험도 계산용 데이터 읽기
final_df = pd.read_csv(SPATIAL_OUTPUT_PATH, nrows=nrows)

print("\n최종 위험도 계산 시작")
print(final_df.shape)

# 최종 위험도 계산
final_df["final_risk_score"] = (
    final_df["weather_risk_score"]
    + final_df["spatial_risk_score"]
)

# 위험 여부 판단 기준 설정
decision_threshold = final_df["final_risk_score"].median()

# 위험 여부 0/1 생성
final_df["decision"] = (
    final_df["final_risk_score"] >= decision_threshold
).astype(int)

# 위험도 4등급 생성
final_df["risk_level"] = pd.qcut(
    final_df["final_risk_score"],
    q=4,
    labels=[
        "4(낮음)",
        "3(보통)",
        "2(높음)",
        "1(매우높음)"
    ]
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

# 결과 확인
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