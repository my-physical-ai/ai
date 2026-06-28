# ============================================================
# 파일명: services/zone_temperature.py
# 설명: SAM3 마스크를 상/중/하 3등분하고 각 영역 온도 통계 측정 + 시각화
# 변경 이력:
#   - [2026-06-22] 최초 생성: 마스크 3등분 + raw 배열 온도 통계 (AI 불필요)
# ============================================================

import cv2                                         # 시각화
import numpy as np                                # 배열 연산

ZONE_NAMES = ["상(上)", "중(中)", "하(下)"]       # 상/중/하 라벨


def measure_three_zones(temp_c, mask, box):
    """SAM3 마스크를 상/중/하 3등분하고 각 영역 온도 통계를 낸다.

    Args:
        temp_c: 온도 배열 (H, W) float32 섭씨
        mask: SAM3 정밀 마스크 (H, W) bool
        box: [x1, y1, x2, y2] — 3등분 기준 높이 범위

    Returns:
        zones: [{"name","min","max","mean","pixels"}, ...] 상/중/하 3개
    """
    x1, y1, x2, y2 = box
    third = (y2 - y1) / 3.0                         # 1/3 높이
    zones = []

    for i, name in enumerate(ZONE_NAMES):
        zy1 = int(y1 + third * i)                   # 구간 시작 y
        zy2 = int(y1 + third * (i + 1))             # 구간 끝 y

        # 3등분 띠 ∩ SAM3 마스크 (배경 제외가 SAM3의 핵심 가치)
        zone_mask = np.zeros_like(mask)             # 빈 마스크
        zone_mask[zy1:zy2, :] = mask[zy1:zy2, :]    # 해당 띠만 살림

        temps = temp_c[zone_mask]                   # 마스크 내부 온도만 추출
        temps = temps[temps > -50]                  # 무효값 제외

        if len(temps) > 0:
            zones.append({
                "name": name,
                "min": float(np.min(temps)),        # 최저 온도
                "max": float(np.max(temps)),        # 최고 온도
                "mean": float(np.mean(temps)),      # 평균 온도
                "pixels": int(len(temps)),          # 측정 픽셀 수
            })
        else:
            zones.append({"name": name, "min": None, "max": None,
                          "mean": None, "pixels": 0})
    return zones


def draw_zones(vis_rgb, mask, box, zones):
    """3등분 경계선 + SAM3 외곽선 + 각 구간 온도를 영상에 그린다."""
    annotated = vis_rgb.copy()
    x1, y1, x2, y2 = box

    # SAM3 마스크 외곽선 (초록) — 정밀 분할 강조
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

    third = (y2 - y1) / 3.0                          # 3등분 높이
    for i, z in enumerate(zones):
        zy = int(y1 + third * (i + 1))               # 경계선 y
        cv2.line(annotated, (x1, zy), (x2, zy), (255, 255, 255), 1)  # 흰색 경계

        if z["mean"] is not None:
            txt = f"{z['name']} {z['mean']:.1f}C (max {z['max']:.1f})"
        else:
            txt = f"{z['name']} N/A"
        ty = int(y1 + third * i + 18)                # 텍스트 y
        cv2.putText(annotated, txt, (x1 + 4, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return annotated
