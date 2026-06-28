# ============================================================
# 파일명: zone_measure.py
# 설명: 박스 내 재정규화로 압축된 상대지수를 0~100으로 펴서 부위 차이를 또렷하게
# 문제: 사람 부위가 inferno 높은 인덱스(60~70)에 몰려 차이가 안 보임
# 해결: 박스 안 픽셀 분포(5~95 백분위)를 0~100으로 재정규화
# 버전: v2026-06-23 (상대지수 압축 해결판)
# ============================================================

import numpy as np                                      # 배열 연산


def measure_zones_renorm(level, box, zone_names, warm_pct=70):
    """박스 내 재정규화로 상/중/하 3등분 상대지수를 펴서 측정한다."""
    Lh, Lw = level.shape[:2]                            # 온도 배열 크기
    x1, y1, x2, y2 = box[:4]                            # 박스 좌표
    x1, x2 = max(0, x1), min(Lw, x2)                    # x 범위 제한
    y1, y2 = max(0, y1), min(Lh, y2)                    # y 범위 제한

    # ★핵심: 박스 전체 픽셀의 5~95 백분위를 0~100 기준으로 잡기
    box_region = level[y1:y2, x1:x2]                    # 박스 전체 영역
    if box_region.size < 10:                            # 너무 작으면 측정 불가
        return [{"name": nm, "min": None, "max": None, "mean": None,
                 "y1": y1, "y2": y2} for nm in zone_names], (x1, y1, x2, y2)
    # 따뜻한 부위(사람)만 골라 재정규화 기준 산출 (차가운 배경 제외)
    warm_mask = box_region >= np.percentile(box_region, warm_pct)  # 따뜻한 픽셀
    warm_vals = box_region[warm_mask]                   # 사람 영역 값
    lo = np.percentile(warm_vals, 5)                    # 하위 5% (펴기 기준 최저)
    hi = np.percentile(warm_vals, 95)                   # 상위 95% (펴기 기준 최고)
    span = max(hi - lo, 1e-3)                           # 0 나눗셈 방지

    # 출력 범위 15~85 — 양 끝이 0/100 극단으로 가지 않게 완만하게 펴기
    OUT_LO, OUT_HI = 15.0, 85.0                         # 완만한 출력 범위

    def renorm(v):
        """박스 내 기준(lo~hi)으로 값을 15~85로 완만하게 재정규화한다."""
        scaled = (v - lo) / span * (OUT_HI - OUT_LO) + OUT_LO  # 15~85로 매핑
        return float(np.clip(scaled, 0, 100))           # 0~100 안전 클립

    third = (y2 - y1) / 3.0                             # 3등분 높이
    zones = []                                           # 구역별 결과
    for i, nm in enumerate(zone_names):                 # 상/중/하 각 구역
        zy1 = max(0, int(y1 + third * i))               # 구역 상단
        zy2 = min(Lh, int(y1 + third * (i + 1)))        # 구역 하단
        region = level[zy1:zy2, x1:x2]                  # 구역 영역
        if region.size > 0:                             # 영역 있으면
            flat = region.flatten()                     # 1차원
            warm_thr = np.percentile(flat, warm_pct)    # 구역 내 따뜻한 경계
            warm = flat[flat >= warm_thr]               # 따뜻한 픽셀
            vals = warm if len(warm) >= 10 else flat    # 적으면 전체 사용
            # 재정규화 적용 — 압축된 값이 0~100으로 펴짐
            zones.append({"name": nm,
                          "min": round(renorm(vals.min())),   # 펴진 최소
                          "max": round(renorm(vals.max())),   # 펴진 최대
                          "mean": round(renorm(vals.mean())), # 펴진 평균
                          "y1": zy1, "y2": zy2})
        else:                                           # 빈 영역
            zones.append({"name": nm, "min": None, "max": None,
                          "mean": None, "y1": zy1, "y2": zy2})
    return zones, (x1, y1, x2, y2)                       # 펴진 구역 결과 반환


def measure_zones_mask(level, box, mask, zone_names):
    """SAM3 마스크로 사람 픽셀만 골라 3등분 상대지수를 측정한다 (배경 제외)."""
    Lh, Lw = level.shape[:2]                            # 온도 배열 크기
    x1, y1, x2, y2 = box[:4]                            # 박스 좌표
    x1, x2 = max(0, x1), min(Lw, x2)                    # x 범위 제한
    y1, y2 = max(0, y1), min(Lh, y2)                    # y 범위 제한

    # 마스크 안(사람) 픽셀만 재정규화 기준으로 사용
    person_vals = level[mask]                           # 마스크 True인 픽셀들
    if person_vals.size < 10:                           # 너무 적으면 박스 방식 폴백
        return measure_zones_renorm(level, box, zone_names)
    lo = np.percentile(person_vals, 5)                  # 사람 영역 하위 5%
    hi = np.percentile(person_vals, 95)                 # 사람 영역 상위 95%
    span = max(hi - lo, 1e-3)                           # 0 나눗셈 방지

    OUT_LO, OUT_HI = 15.0, 85.0                         # 완만한 출력 범위

    def renorm(v):
        """사람 영역 기준으로 값을 15~85로 재정규화한다."""
        scaled = (v - lo) / span * (OUT_HI - OUT_LO) + OUT_LO  # 15~85 매핑
        return float(np.clip(scaled, 0, 100))           # 0~100 안전 클립

    third = (y2 - y1) / 3.0                             # 3등분 높이
    zones = []                                           # 구역별 결과
    for i, nm in enumerate(zone_names):                 # 상/중/하 각 구역
        zy1 = max(0, int(y1 + third * i))               # 구역 상단
        zy2 = min(Lh, int(y1 + third * (i + 1)))        # 구역 하단
        # 해당 구역에서 마스크 안(사람) 픽셀만 추출
        zone_level = level[zy1:zy2, x1:x2]              # 구역 온도
        zone_mask = mask[zy1:zy2, x1:x2]               # 구역 마스크
        vals = zone_level[zone_mask]                    # 사람 픽셀만
        if vals.size > 0:                               # 사람 픽셀 있으면
            zones.append({"name": nm,
                          "min": round(renorm(vals.min())),   # 펴진 최소
                          "max": round(renorm(vals.max())),   # 펴진 최대
                          "mean": round(renorm(vals.mean())), # 펴진 평균
                          "y1": zy1, "y2": zy2})
        else:                                           # 사람 픽셀 없으면
            zones.append({"name": nm, "min": None, "max": None,
                          "mean": None, "y1": zy1, "y2": zy2})
    return zones, (x1, y1, x2, y2)                       # 마스크 기반 결과 반환
