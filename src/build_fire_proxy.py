"""
================================================================================
 산불 이력 → 전주 proxy 위험신호 생성 파이프라인
 (safemap 산불발생이력 IF_0088 공식 명세 반영본)
================================================================================
 흐름:
   1) safemap IF_0088 호출 → 산불 좌표(x,y)·원인·발생연도 수집
   2) 좌표계 변환: EPSG:3857(Web Mercator) → EPSG:4326(위경도)  [검증 완료]
   3) 전주(pole) 데이터 로드 (4326)
   4) 전주별 "반경 N km 내 산불 건수 / 피해면적 합" 계산 → proxy 신호
   5) 결과 저장 (규칙기반 점수의 '사후 검증'용 권장)

 실행 전:
   - SERVICE_KEY 에 발급받은 인증키 입력
   - pip install requests pandas scikit-learn
   - POLE_PATH 에 대회 전주 파일 경로 지정 (pole_id, lon, lat 컬럼)
================================================================================
"""

import math
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

# ------------------------------------------------------------------ 설정
SERVICE_KEY = "UAC5QGW7-UAC5-UAC5-UAC5-UAC5QGW787"   # ← 본인 인증키
API_URL = "https://safemap.go.kr/openapi2/IF_0088"   # 공식 XML 호출 URL
POLE_PATH = "data/poles.csv"          # 대회 제공 전주 파일 (pole_id, lon, lat)
RADIUS_KM = 3.0                       # 산불 탐색 반경(km) — 민감도 분석 대상
OUT_PATH = "data/pole_fire_proxy.csv"


# ------------------------------------------------------------------ 1) API 호출
def fetch_fire_history(max_pages=50, rows_per_page=1000):
    records = []
    for page in range(1, max_pages + 1):
        params = {
            "serviceKey": SERVICE_KEY,
            "pageNo": page,
            "numOfRows": rows_per_page,
            "returnType": "xml",      # 기본값이 JSON이므로 명시 필요
        }
        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[page {page}] 요청 실패: {e}")
            break

        if page == 1:
            print("=" * 60)
            print("요청 URL:", resp.url)
            print("상태코드:", resp.status_code)
            print("응답 앞부분 ↓↓↓")
            print(resp.text[:600])
            print("=" * 60)

        if resp.text.lstrip()[:1] != "<":
            print("⚠️ XML이 아닌 응답 (위 응답 앞부분 확인). 중단.")
            break

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            print(f"⚠️ XML 파싱 실패: {e}")
            break

        items = root.findall(".//item")
        if not items:
            # item 태그가 아닐 경우 대비: 루트 구조 일부 출력
            if page == 1:
                print("item 태그 없음. 자식 태그명 일부:",
                      [c.tag for c in list(root.iter())][:20])
            print(f"[page {page}] item 없음 → 종료")
            break

        for it in items:
            def g(tag):
                el = it.find(tag)
                return el.text if el is not None else None
            # 강원도(42)만 수집 — 전주가 강원에만 있으므로
            if g("ctprvn_cd") != "42":
                continue
            records.append({
                "x": g("x"), "y": g("y"),
                "adres": g("adres"), "resn": g("resn"),
                "ar": g("ar"), "amount": g("amount"),
                "occu_year": g("occu_year"), "occu_date": g("occu_date"),
                "ctprvn_cd": g("ctprvn_cd"), "objt_id": g("objt_id"),
            })
        print(f"[page {page}] 누적 {len(records)}건")
        time.sleep(0.05)

    df = pd.DataFrame(records)
    for c in ["x", "y", "ar", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    print(f"총 산불 이력: {len(df)}건")
    return df


# ------------------------------------------------------------------ 2) 좌표 변환
def webmercator_to_wgs84(x, y):
    """EPSG:3857 → EPSG:4326 (직접 계산, 검증 완료)."""
    R = 6378137.0
    lon = (x / R) * 180.0 / math.pi
    lat = (2 * math.atan(math.exp(y / R)) - math.pi / 2) * 180.0 / math.pi
    return lon, lat


def add_lonlat(df):
    lonlat = df.apply(lambda r: webmercator_to_wgs84(r["x"], r["y"]), axis=1)
    df["lon"] = lonlat.map(lambda t: t[0])
    df["lat"] = lonlat.map(lambda t: t[1])
    bad = df[(df.lon < 124) | (df.lon > 132) | (df.lat < 33) | (df.lat > 39)]
    if len(bad):
        print(f"⚠️  한국 범위 벗어난 좌표 {len(bad)}건 — 좌표계 재확인 필요")
    else:
        print(f"좌표 변환 OK (전부 한국 범위 내). 예: "
              f"경도 {df.lon.iloc[0]:.4f}, 위도 {df.lat.iloc[0]:.4f}")
    return df


# ------------------------------------------------------------------ 3~4) 공간조인
def build_proxy(poles, fires, radius_km=RADIUS_KM, years=None, prefix=None):
    """전신주 주변 산불 proxy 컬럼 생성."""
    import numpy as np
    from sklearn.neighbors import BallTree

    fires = fires.copy()
    if years is not None:
        if "occu_year" not in fires.columns:
            raise ValueError("years 옵션을 쓰려면 fires에 occu_year 컬럼이 필요합니다.")
        # 산불 연도 필터링
        fires["occu_year"] = pd.to_numeric(fires["occu_year"], errors="coerce")
        fires = fires[fires["occu_year"].isin(years)].copy()

    # 출력 컬럼명 설정
    if prefix is None:
        count_column = f"fire_cnt_{int(radius_km)}km"
        area_column = f"fire_area_{int(radius_km)}km"
        proxy_column = "fire_proxy"
    else:
        count_column = f"{prefix}_fire_count_{int(radius_km)}km"
        area_column = f"{prefix}_fire_area_{int(radius_km)}km"
        proxy_column = f"{prefix}_fire_proxy_{int(radius_km)}km"

    poles = poles.copy()
    if fires.empty:
        poles[count_column] = 0
        poles[area_column] = 0.0
        poles[proxy_column] = 0
        return poles

    # 위경도 기반 반경 검색
    fire_rad = np.radians(fires[["lat", "lon"]].values)
    pole_rad = np.radians(poles[["lat", "lon"]].values)
    tree = BallTree(fire_rad, metric="haversine")

    r = radius_km / 6371.0
    idx_within = tree.query_radius(pole_rad, r=r)

    counts, area_sum = [], []
    for idxs in idx_within:
        counts.append(len(idxs))
        if len(idxs) and "ar" in fires.columns:
            area_sum.append(float(fires.iloc[idxs]["ar"].fillna(0).sum()))
        else:
            area_sum.append(0.0)

    poles[count_column] = counts
    poles[area_column] = area_sum
    poles[proxy_column] = (poles[count_column] > 0).astype(int)
    return poles


# ------------------------------------------------------------------ main
def main():
    fires = fetch_fire_history()
    if len(fires) == 0:
        print("산불 데이터를 못 받았습니다. 위 로그 확인.")
        return
    fires = add_lonlat(fires)

    fires.to_csv("data/fire_history.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 산불 이력 저장 완료: data/fire_history.csv ({len(fires)}건)")

if __name__ == "__main__":
    main()
