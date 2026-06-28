# ============================================================
# 파일명: thermal_level.py
# 설명: FLIR ONE RGB 의사색상 → 정확한 상대온도 지수(0~100) 역변환 모듈
# 핵심: BGR2GRAY는 노랑>주황 역전이 발생하므로 inferno 역변환 LUT 사용
# 버전: v2026-06-23 (온도 순위 역전 해결판)
# ============================================================

import cv2                                              # 영상 처리
import numpy as np                                      # 배열 연산


def _build_inferno_lut():
    """inferno 컬러맵의 BGR→인덱스(0~255) 역변환 LUT를 1회 생성한다."""
    idx = np.arange(256, dtype=np.uint8).reshape(256, 1)   # 0~255 인덱스
    bgr = cv2.applyColorMap(idx, cv2.COLORMAP_INFERNO)     # 인덱스 → inferno BGR
    bgr = bgr.reshape(256, 3).astype(np.int16)             # (256,3) 정수 배열
    return bgr                                             # 팔레트 BGR 테이블 반환


# 모듈 로드 시 LUT 1회 계산 (매 프레임 재계산 방지)
_INFERNO_BGR = _build_inferno_lut()                        # inferno 팔레트 테이블


def rgb_to_level_inferno(vis_bgr):
    """inferno RGB 프레임을 팔레트 인덱스 기반 상대온도 지수(0~100)로 역변환한다."""
    h, w = vis_bgr.shape[:2]                               # 프레임 크기
    flat = vis_bgr.reshape(-1, 3).astype(np.float32)       # (H*W, 3) float (오버플로 방지)
    pal = _INFERNO_BGR.astype(np.float32)                  # 팔레트도 float로
    # 각 픽셀과 256개 팔레트 색의 거리 → 가장 가까운 팔레트 인덱스 찾기
    # (N,1,3) - (256,3) 브로드캐스팅으로 색거리 계산
    diff = flat[:, None, :] - pal[None, :, :]              # 색 차이 (N,256,3)
    dist = np.einsum("nkc,nkc->nk", diff, diff)            # 제곱거리 (N,256)
    nearest = np.argmin(dist, axis=1)                      # 최근접 팔레트 인덱스 (N,)
    level = nearest.reshape(h, w).astype(np.float32) / 255.0 * 100.0  # 0~100 지수
    return level                                           # 온도 순서 보존된 지수 반환


def rgb_to_level_fast(vis_bgr):
    """빠른 근사판 — R채널 우선 + 거리 LUT 없이 inferno 단조성 근사 (실시간용)."""
    b, g, r = cv2.split(vis_bgr.astype(np.float32))        # 채널 분리
    # inferno는 저온(보라:B우세)→고온(노랑흰:R,G우세)이므로 가중 합성
    # R에 가장 큰 가중, 어두운 보라(저온)는 낮게 — 노랑>주황 역전 방지
    level = r * 0.5 + g * 0.4 - b * 0.1                    # inferno 단조 근사
    level = np.clip(level, 0, 255) / 255.0 * 100.0         # 0~100 정규화
    return level.astype(np.float32)                        # 근사 지수 반환


def rgb_to_level_grayscale(vis_bgr):
    """회색조(Grayscale.raw) 팔레트 전용 — 밝기를 그대로 상대지수(0~100)로 변환한다."""
    # 회색조는 R=G=B이고 밝기가 곧 온도 순위 → 컬러 역변환 불필요, 밝기만 사용
    if vis_bgr.ndim == 3:                                 # 3채널 회색조면
        gray = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)  # 한 채널로 (R=G=B라 손실 없음)
    else:                                                 # 이미 단일채널이면
        gray = vis_bgr                                    # 그대로 사용
    level = gray.astype(np.float32) / 255.0 * 100.0       # 밝기 → 0~100 상대지수
    return level                                          # 온도순위 보존된 지수 반환
