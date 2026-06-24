# NUC LOCAL 전용 Flask 서버 — RealSense D435i/D455 직결 미션 1~7
# [2026-02-21 생성] Type A 전용, Pi/ZeroMQ 코드 완전 제거

import time         # 시간 측정 및 FPS 제어용
import threading    # RealSense 접근 동기화용 Lock

import cv2                      # 영상 처리 및 JPEG 인코딩
import numpy as np              # Depth 배열 수치 연산
import pyrealsense2 as rs       # RealSense SDK
from flask import (             # 웹서버 프레임워크
    Flask, render_template, Response, jsonify, request
)

# ============================================================
# ★ 사용자 환경에 맞게 수정할 설정값들
# ============================================================
FLASK_HOST = "0.0.0.0"   # 모든 네트워크에서 접속 허용 (localhost만 원하면 "127.0.0.1")
FLASK_PORT = 5000         # Flask 웹서버 포트 번호
CAMERA_W = 640            # RealSense Color/Depth 해상도 가로 (픽셀)
CAMERA_H = 480            # RealSense Color/Depth 해상도 세로 (픽셀)
CAMERA_FPS = 30           # RealSense 프레임 레이트 (초당 프레임 수)

# ============================================================
# Flask 앱 + 전역 상태 변수
# ============================================================
app = Flask(__name__)     # Flask 애플리케이션 인스턴스 생성

# --- RealSense 공유 상태 ---
rs_pipeline = None        # pyrealsense2 파이프라인 (카메라 스트림 관리)
rs_align = None           # Depth→Color 좌표계 정렬 객체 (필수!)
rs_depth_scale = 0.001    # Depth 스케일 값 (raw값 × scale = 미터)
rs_intrinsics = None      # 카메라 내부 파라미터 (fx, fy, cx, cy)
rs_lock = threading.Lock()  # 멀티스레드 RealSense 접근 동기화
# [2026-02-22 추가] 미션 전환 시 이전 generator를 종료하기 위한 상태 변수
active_mission = 0            # 현재 활성 미션 번호 (0=없음)

# --- RealSense Depth 필터 (미션 6에서 사용) ---
spatial_filter = rs.spatial_filter()                            # 공간 필터 생성
spatial_filter.set_option(rs.option.filter_magnitude, 2)        # 필터 강도 2 (1~5)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)   # 공간 평활 계수 (0~1)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)    # 에지 보존 임계값 (1~50)

temporal_filter = rs.temporal_filter()                          # 시간 필터 생성
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)  # 시간 평활 계수 (0~1)
temporal_filter.set_option(rs.option.filter_smooth_delta, 20)   # 변화 임계값 (1~100)

hole_filter = rs.hole_filling_filter()                          # 빈 영역 채우기 필터

# --- 미션별 인터랙션 상태 (UI와 공유) ---
state = {
    "click_point": None,            # (x, y) 마지막 클릭 좌표
    "click_depth": 0.0,             # 마지막 클릭 지점의 거리 (미터)
    "click_3d": None,               # (X, Y, Z) 3D 실제 좌표 (미터)
    "click_pixel": None,            # (u, v) 마지막 클릭 픽셀 좌표
    "click_history": [],            # 클릭 이력 리스트 (최대 5개)
    "colormap": cv2.COLORMAP_JET,   # 현재 Depth 컬러맵 ID
    "colormap_name": "JET",         # 현재 컬러맵 이름 (화면 표시용)
    "roi_start": None,              # ROI 드래그 시작점 [x, y]
    "roi_end": None,                # ROI 드래그 끝점 [x, y]
    "roi_stats": None,              # ROI 영역 통계 (mean, min, max, std)
    "alert_zone_pct": 30,           # 근접 경고 ROI 비율 (%, 슬라이더)
    "alert_dist_m": 0.5,            # 근접 경고 임계 거리 (미터, 슬라이더)
}

# --- 기준 프레임 저장소 (4-1, 5-1, 6-1 공장 미션용) ---
# [2026-02-22 추가] 배경/양품/빈팔레트 Depth를 캡처하여 비교 기준으로 사용
ref_frames = {
    "bg_depth": None,               # 미션 41: 빈 컨베이어 배경 Depth (float64)
    "golden_depth": None,           # 미션 51: 양품 Golden Sample Depth (float64)
    "empty_depth": None,            # 미션 61: 빈 팔레트 Depth (float64)
    "spec_height_mm": 0,             # 미션 41: 규격 높이 (mm) — 0=미설정, >0=검사 모드
    "spec_tolerance_mm": 2.0,       # 미션 41: 허용 오차 (mm)
    "defect_threshold_mm": 0.5,     # 미션 51: 결함 임계값 (mm)
    "max_stack_mm": 500.0,          # 미션 61: 최대 적재 높이 (mm)
}


# ============================================================
# 1. RealSense 초기화 / 해제 / 프레임 획득
# ============================================================
def init_realsense():
    """RealSense 카메라를 초기화하고 스트림을 시작한다."""
    global rs_pipeline, rs_align, rs_depth_scale, rs_intrinsics

    rs_pipeline = rs.pipeline()           # 파이프라인 생성
    config = rs.config()                  # 스트림 설정 객체 생성

    # Color 스트림 활성화 (BGR 포맷, OpenCV 호환)
    config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)
    # Depth 스트림 활성화 (16비트 정수, 밀리미터 단위)
    config.enable_stream(rs.stream.depth, CAMERA_W, CAMERA_H, rs.format.z16, CAMERA_FPS)

    profile = rs_pipeline.start(config)   # 스트림 시작 → 프로파일 반환

    # [2026-02-21 추가] Depth를 Color 좌표계에 정렬 (클릭 좌표 정확도 보장)
    rs_align = rs.align(rs.stream.color)

    # Depth 스케일 값 읽기 (장치마다 다름, 보통 0.001 = 1mm 단위)
    depth_sensor = profile.get_device().first_depth_sensor()
    rs_depth_scale = depth_sensor.get_depth_scale()

    # 카메라 내부 파라미터 읽기 (역투영 공식에 사용)
    stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    rs_intrinsics = stream_profile.get_intrinsics()

    # 초기화 결과 출력
    print(f"✅ RealSense 초기화 완료")
    print(f"   depth_scale = {rs_depth_scale:.6f}")
    print(f"   fx={rs_intrinsics.fx:.1f}  fy={rs_intrinsics.fy:.1f}")
    print(f"   cx={rs_intrinsics.ppx:.1f}  cy={rs_intrinsics.ppy:.1f}")


def stop_realsense():
    """RealSense 파이프라인을 안전하게 정지한다."""
    global rs_pipeline
    if rs_pipeline is not None:
        try:
            rs_pipeline.stop()            # 스트림 정지
        except Exception:
            pass                          # 이미 정지된 경우 무시
        rs_pipeline = None
        print("🛑 RealSense 정지 완료")


def get_frames(apply_filter=False):
    """RealSense에서 Color + Depth 프레임을 1세트 가져온다.

    Args:
        apply_filter: True면 Spatial+Temporal+HoleFilling 필터 적용

    Returns:
        (color_image, depth_frame, depth_array) 또는 실패 시 (None, None, None)
    """
    if rs_pipeline is None:               # 카메라 미초기화 시 즉시 반환
        return None, None, None
    try:
        with rs_lock:                     # 스레드 안전 — 동시 접근 방지
            # 프레임 대기 (최대 1초)
            frames = rs_pipeline.wait_for_frames(timeout_ms=1000)
            # Depth→Color 좌표계 정렬 적용 (필수!)
            aligned = rs_align.process(frames)
            color_frame = aligned.get_color_frame()   # Color 프레임 추출
            depth_frame = aligned.get_depth_frame()   # Depth 프레임 추출

            if not color_frame or not depth_frame:    # 프레임 없으면 반환
                return None, None, None

            # [2026-02-21 추가] 필터 ON 시 Spatial→Temporal→HoleFill 순서 적용
            if apply_filter:
                depth_frame = spatial_filter.process(depth_frame)
                depth_frame = temporal_filter.process(depth_frame)
                depth_frame = hole_filter.process(depth_frame)

            # NumPy 배열로 변환 (OpenCV 처리용)
            color_image = np.asanyarray(color_frame.get_data())   # (H,W,3) BGR
            depth_array = np.asanyarray(depth_frame.get_data())   # (H,W) uint16
            return color_image, depth_frame, depth_array
    except Exception:
        return None, None, None           # 에러 발생 시 안전 반환


# ============================================================
# 2. 미션별 프레임 처리 함수 (미션 1~7)
# ============================================================

def process_m1(color, depth_frame, depth_array):
    """미션 1: RGB + Depth 듀얼 뷰 — 좌우 나란히 표시."""
    # Depth 배열을 0~255 범위로 정규화
    depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # JET 컬러맵 적용 (가까움=빨강, 멀리=파랑)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    # 좌(Color) + 우(Depth) 수평 결합 → 1280×480 이미지
    combined = np.hstack([color, depth_colored])
    # 라벨 표시
    cv2.putText(combined, "RGB", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, "DEPTH", (CAMERA_W + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return combined


def process_m2(color, depth_frame, depth_array):
    """미션 2: 클릭 거리 측정 — 화면 클릭 시 해당 지점의 실제 거리(cm) 표시."""
    display = color.copy()                # 원본 보존을 위해 복사
    h, w = display.shape[:2]              # 프레임 높이, 너비

    # 화면 중심 실시간 거리 표시
    center_d = depth_frame.get_distance(w // 2, h // 2) if depth_frame else 0
    cv2.drawMarker(display, (w // 2, h // 2), (0, 200, 200),
                   cv2.MARKER_CROSS, 15, 1)   # 중심 십자 마커
    if center_d > 0:
        cv2.putText(display, f"Center: {center_d:.2f}m",
                    (w // 2 + 15, h // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

    # 클릭 이력 표시 (최근 5개, 점점 밝아지는 효과)
    for i, (cx, cy, cd) in enumerate(state.get("click_history", [])):
        alpha = (i + 1) / max(len(state["click_history"]), 1)   # 투명도 비율
        c = (0, int(255 * alpha), 0)                            # 밝기 증가
        cv2.circle(display, (cx, cy), 6, c, -1)                 # 채운 원
        label = f"{cd:.2f}m" if cd > 0 else "N/A"               # 거리 텍스트
        cv2.putText(display, label, (cx + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    # 현재 클릭 포인트 십자선
    if state["click_point"]:
        px, py = state["click_point"]
        cv2.line(display, (px - 20, py), (px + 20, py), (0, 255, 255), 1)
        cv2.line(display, (px, py - 20), (px, py + 20), (0, 255, 255), 1)

    # 클릭 거리 대형 표시
    if state["click_point"] and state["click_depth"] > 0:
        cv2.putText(display, f"{state['click_depth']*100:.1f} cm",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # 미션 제목
    cv2.putText(display, "RANGE FINDER | Click to measure",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2)
    return display


def process_m3(color, depth_frame, depth_array):
    """미션 3: 2D→3D 좌표 변환 — 역투영 공식으로 (X,Y,Z) 산출."""
    display = color.copy()                # 원본 보존

    # 클릭 3D 좌표 시각화
    if state["click_pixel"] and state["click_3d"]:
        px, py = state["click_pixel"]     # 클릭 픽셀 좌표
        X, Y, Z = state["click_3d"]       # 실제 3D 좌표 (미터)
        # 십자선 + 원 마커
        cv2.drawMarker(display, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.circle(display, (px, py), 6, (0, 255, 0), -1)
        # X, Y, Z 값 텍스트 (cm 단위)
        cv2.putText(display, f"X: {X*100:+.1f}cm", (px+15, py-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(display, f"Y: {Y*100:+.1f}cm", (px+15, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        cv2.putText(display, f"Z: {Z*100:.1f}cm", (px+15, py+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 0), 1)

    # 카메라 내부 파라미터 표시 (우측 상단)
    if rs_intrinsics:
        y0 = 50
        cv2.putText(display, "INTRINSICS", (480, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 180), 1)
        params = [("fx", rs_intrinsics.fx), ("fy", rs_intrinsics.fy),
                  ("cx", rs_intrinsics.ppx), ("cy", rs_intrinsics.ppy)]
        for name, val in params:
            y0 += 18                      # 줄 간격
            cv2.putText(display, f"{name}: {val:.1f}", (480, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 150), 1)

    # 역투영 공식 표시 (하단)
    cv2.putText(display, "X=(u-cx)*Z/fx  Y=(v-cy)*Z/fy",
                (10, CAMERA_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 180, 180), 1)
    # 미션 제목
    cv2.putText(display, "3D COORDINATE | Click for XYZ",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (213, 0, 249), 2)
    return display


def process_m4(color, depth_frame, depth_array):
    """미션 4: Depth 컬러맵 변경 — JET/TURBO/BONE/HOT 전환."""
    # Depth 정규화
    depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # 현재 선택된 컬러맵 적용
    depth_colored = cv2.applyColorMap(depth_norm, state["colormap"])
    # 좌우 결합 (Color + Depth)
    combined = np.hstack([color, depth_colored])
    # 라벨 표시
    cv2.putText(combined, "RGB", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (213, 0, 249), 2)
    cv2.putText(combined, f"DEPTH [{state['colormap_name']}]",
                (CAMERA_W + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # 조작 안내 (하단)
    cv2.putText(combined, "1:JET  2:TURBO  3:BONE  4:HOT",
                (CAMERA_W + 10, CAMERA_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return combined


def process_m5(color, depth_frame, depth_array):
    """미션 5: ROI 드래그 → Depth 영역 통계 (mean/min/max/std)."""
    # Depth를 JET 컬러맵으로 시각화
    depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    display = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # ROI 드래그 중인 박스 표시 (노란색)
    if state["roi_start"] and state["roi_end"]:
        cv2.rectangle(display, tuple(state["roi_start"]),
                      tuple(state["roi_end"]), (0, 255, 255), 2)

    # ROI 통계 패널 표시 (초록색)
    if state["roi_stats"]:
        s = state["roi_stats"]            # 통계 딕셔너리
        x1, y1, x2, y2 = s["roi"]        # ROI 좌표
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 확정 ROI
        # 통계 패널 배경 (우측 상단)
        px, py0 = 440, 60
        cv2.rectangle(display, (px - 10, py0 - 25), (635, py0 + 120), (0, 0, 0), -1)
        cv2.rectangle(display, (px - 10, py0 - 25), (635, py0 + 120), (0, 255, 0), 1)
        cv2.putText(display, "ROI STATS", (px, py0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        # 통계값 텍스트 (cm 단위로 변환)
        lines = [
            f"Mean:  {s['mean']*100:.1f} cm",
            f"Min:   {s['min']*100:.1f} cm",
            f"Max:   {s['max']*100:.1f} cm",
            f"Std:   {s['std']*100:.2f} cm",
            f"Valid: {s['valid_pct']:.1f}%",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(display, txt, (px, py0 + 15 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

    # 미션 제목
    cv2.putText(display, "ZONE RECON | Drag to select ROI",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (213, 0, 249), 2)
    return display


def process_m6(color, depth_frame, depth_array):
    """미션 6: 필터 비교 — RAW vs FILTERED 나란히 표시."""
    # 필터 적용된 Depth를 별도로 가져옴
    _, _, depth_filtered = get_frames(apply_filter=True)
    if depth_filtered is None:            # 실패 시 원본 사용
        depth_filtered = depth_array

    # RAW Depth 시각화
    raw_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_JET)

    # FILTERED Depth 시각화
    filt_norm = cv2.normalize(depth_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    filt_colored = cv2.applyColorMap(filt_norm, cv2.COLORMAP_JET)

    # 좌(RAW) + 우(FILTERED) 결합
    combined = np.hstack([raw_colored, filt_colored])
    # 라벨 표시
    cv2.putText(combined, "RAW (No Filter)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2)
    cv2.putText(combined, "FILTERED (Spatial+Temporal+HoleFill)",
                (CAMERA_W + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 2)
    return combined


def process_m7(color, depth_frame, depth_array):
    """미션 7: 근접 경고 — ROI 영역 최소 거리 감지 → 3단계 알림."""
    display = color.copy()                # 원본 보존
    h, w = display.shape[:2]              # 프레임 크기

    # --- 감지 영역(ROI) 계산 (슬라이더 제어) ---
    roi_pct = state.get("alert_zone_pct", 30)  # 화면 대비 ROI 비율 (%)
    rh = int(h * roi_pct / 100)                # ROI 높이 (픽셀)
    rw = int(w * roi_pct / 100)                # ROI 너비 (픽셀)
    y1 = (h - rh) // 2                        # ROI 상단 y
    y2 = (h + rh) // 2                        # ROI 하단 y
    x1 = (w - rw) // 2                        # ROI 좌측 x
    x2 = (w + rw) // 2                        # ROI 우측 x

    # --- ROI 내 최소 거리 산출 ---
    roi = depth_array[y1:y2, x1:x2].astype(np.float64) * rs_depth_scale  # 미터 변환
    valid = roi[roi > 0.1]                     # 0값(측정불가) 제거 (0.1m 이상만)
    min_dist = float(np.min(valid)) if len(valid) > 0 else 0.0           # 최소 거리
    alert_dist = state.get("alert_dist_m", 0.5)  # 경고 임계 거리 (슬라이더)
    is_warning = 0 < min_dist < alert_dist        # 경고 조건 판정

    # --- ROI 영역 시각화 ---
    roi_color = (0, 0, 255) if is_warning else (0, 255, 255)  # 위험=빨강, 안전=노랑
    cv2.rectangle(display, (x1, y1), (x2, y2), roi_color, 2)  # ROI 박스
    cv2.drawMarker(display, (w // 2, h // 2), roi_color,
                   cv2.MARKER_CROSS, 25, 2)   # 중심 십자선

    # --- 경고 시각 효과 ---
    if is_warning:
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)  # 빨간 테두리
        # 깜빡임 효과 (4Hz)
        if int(time.time() * 4) % 2 == 0:
            overlay = display.copy()          # 오버레이 복사
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 80), -1)  # 반투명 빨강
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)  # 블렌딩
        # 경고 텍스트 (대형)
        cv2.putText(display, f"WARNING! {min_dist*100:.0f}cm",
                    (w // 2 - 160, h // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif min_dist > 0:
        # 안전 시 현재 거리 표시
        cv2.putText(display, f"{min_dist*100:.1f}cm",
                    (w // 2 + 20, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # --- 상태 정보 패널 (우측 하단) ---
    cv2.rectangle(display, (w - 250, h - 100), (w - 5, h - 5), (0, 0, 0), -1)
    cv2.putText(display, f"Zone: {roi_pct}%  Thresh: {alert_dist*100:.0f}cm",
                (w - 240, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(display, f"Min dist: {min_dist*100:.1f}cm",
                (w - 240, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    st_text = "DANGER" if is_warning else "SAFE"      # 상태 문자열
    st_color = (0, 0, 255) if is_warning else (0, 255, 0)  # 색상
    cv2.putText(display, f"Status: {st_text}",
                (w - 240, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, st_color, 1)

    # 미션 제목
    cv2.putText(display, "PROXIMITY ALERT | Local RealSense",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 23, 68), 2)
    return display


# ============================================================
# 2-1. 공장 적용 미션 (4-1, 5-1, 6-1)
# [2026-02-22 추가] 기준 프레임 대비 실시간 검사
# ============================================================

def process_m41(color, depth_frame, depth_array):
    """미션 4-1: 제품 높이 검사 — 2단계 워크플로우.

    Step 1: 배경 캡처 → Step 2: 양품 놓고 규격 설정 → 이후 자동 합격/불합격
    """
    display = color.copy()
    h, w = display.shape[:2]

    if ref_frames["bg_depth"] is None:
        cv2.putText(display, "Step 1: Press [CAPTURE] to record background",
                    (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(display, "HEIGHT INSPECTOR | No background",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        return display

    # 중앙 60% ROI
    ry1, ry2 = int(h * 0.2), int(h * 0.8)
    rx1, rx2 = int(w * 0.2), int(w * 0.8)
    bg_roi = ref_frames["bg_depth"][ry1:ry2, rx1:rx2]
    now_roi = depth_array[ry1:ry2, rx1:rx2].astype(np.float64)

    # 배경 대비 높이 차분 (mm) — 0~2000mm 범위
    diff_mm = (bg_roi - now_roi) * rs_depth_scale * 1000
    diff_mm = np.clip(diff_mm, 0, 2000)

    mask = diff_mm > 30                       # 30mm 이상 = 제품
    product_pixels = np.sum(mask)
    cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 200, 255), 1)  # ROI 표시

    if product_pixels > 2000:
        height_mm = float(np.percentile(diff_mm[mask], 95))
        spec = ref_frames["spec_height_mm"]
        tol = ref_frames["spec_tolerance_mm"]

        # 제품 영역 오버레이
        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[ry1:ry2, rx1:rx2] = mask

        # [2026-02-22 수정] 규격 미설정(0) → 측정만, 규격 설정됨 → 합격/불합격
        if spec <= 0:
            # === 측정 모드 (규격 미설정) ===
            overlay = display.copy()
            overlay[full_mask] = (200, 180, 0)        # 노란색 = 측정 중
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.putText(display, f"{height_mm:.1f}mm", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
            cv2.putText(display, "MEASURING", (w // 2 - 80, h // 2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display, "Step 2: Press [SET SPEC] with good product",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        else:
            # === 검사 모드 (규격 설정됨) ===
            deviation = abs(height_mm - spec)
            is_pass = deviation <= tol
            overlay = display.copy()
            color_fill = (0, 200, 0) if is_pass else (0, 0, 230)
            overlay[full_mask] = color_fill
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            result_text = "PASS" if is_pass else "FAIL"
            result_color = (0, 255, 0) if is_pass else (0, 0, 255)
            cv2.putText(display, f"{height_mm:.1f}mm", (w // 2 - 80, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, result_color, 4)
            cv2.putText(display, result_text, (w // 2 - 50, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 3)
            cv2.putText(display, f"Spec: {spec:.0f} +/-{tol:.0f}mm  Dev: {deviation:.1f}mm",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            if not is_pass:
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
    else:
        cv2.putText(display, "No product detected",
                    (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    mode = "MEASURING" if ref_frames["spec_height_mm"] <= 0 else f"Spec: {ref_frames['spec_height_mm']:.0f}mm"
    cv2.putText(display, f"HEIGHT INSPECTOR | {mode}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    return display


def process_m51(color, depth_frame, depth_array):
    """미션 5-1: 표면 결함 히트맵 — 양품 대비 편차를 TURBO 컬러맵으로 시각화."""
    h, w = depth_array.shape

    if ref_frames["golden_depth"] is None:
        display = color.copy()
        cv2.putText(display, "Press [CAPTURE] to record golden sample",
                    (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(display, "DEFECT HEATMAP | No golden sample",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)
        return display

    # 양품 대비 편차 (mm)
    diff_mm = np.abs(
        depth_array.astype(np.float64) - ref_frames["golden_depth"]
    ) * rs_depth_scale * 1000
    threshold = ref_frames["defect_threshold_mm"]

    # 편차를 TURBO 컬러맵으로 시각화
    diff_clipped = np.clip(diff_mm, 0, 5.0)                       # 0~5mm 범위로 클리핑
    heatmap_norm = cv2.normalize(diff_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_TURBO)

    # 결함 마스크 (threshold 초과)
    defect_mask = diff_mm > threshold
    defect_pct = np.sum(defect_mask) / defect_mask.size * 100

    # 원본 Color + 히트맵 나란히
    combined = np.hstack([color, heatmap_color])

    # 결함 박스 표시 (Color 쪽에)
    if defect_pct > 0.5:
        mask_u8 = defect_mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 80:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(combined, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(combined, "DEFECT", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 상태 표시
    status = "DEFECT!" if defect_pct > 1.0 else "OK"
    status_color = (0, 0, 255) if defect_pct > 1.0 else (0, 255, 0)
    cv2.putText(combined, f"COLOR", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)
    cv2.putText(combined, f"TURBO HEATMAP | Defect: {defect_pct:.1f}% [{status}]",
                (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 2)
    cv2.putText(combined, f"Threshold: {threshold:.1f}mm",
                (w + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return combined


def process_m61(color, depth_frame, depth_array):
    """미션 6-1: 적재량 모니터링 — 빈 팔레트 대비 채움률(%) 산출."""
    display = color.copy()
    h, w = display.shape[:2]

    if ref_frames["empty_depth"] is None:
        cv2.putText(display, "Press [CAPTURE] to record empty pallet",
                    (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(display, "FILL LEVEL | No empty reference",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 180), 2)
        return display

    # 빈 팔레트 대비 높이 차분 (mm)
    diff_mm = (ref_frames["empty_depth"] - depth_array.astype(np.float64)) * rs_depth_scale * 1000
    diff_mm = np.clip(diff_mm, 0, 2000)       # 0~2000mm 범위로 클리핑

    # [2026-02-22 수정] 면적 기반 채움률 — 20mm 이상 차이나는 픽셀 비율
    # "화면의 몇 %에 물건이 있는가" → 직관적!
    has_stuff = diff_mm > 20                   # 20mm 이상 = 물건 있음
    fill_pct = float(np.sum(has_stuff) / has_stuff.size * 100)
    fill_pct = min(fill_pct, 100.0)

    # 물건 있는 영역의 평균 높이 (정보 표시용)
    if np.sum(has_stuff) > 0:
        avg_h = float(np.mean(diff_mm[has_stuff]))
    else:
        avg_h = 0.0

    # 높이 차분을 컬러 오버레이 (물건 있는 곳만)
    fill_norm = cv2.normalize(diff_mm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    fill_color = cv2.applyColorMap(fill_norm, cv2.COLORMAP_JET)
    # 물건 없는 곳은 원본 유지, 있는 곳만 히트맵
    mask_3ch = np.stack([has_stuff] * 3, axis=-1)
    display = np.where(mask_3ch, cv2.addWeighted(fill_color, 0.5, display, 0.5, 0), display)

    # 채움률 게이지 바 (하단)
    bar_x, bar_y, bar_w, bar_h = 50, h - 60, w - 100, 30
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * fill_pct / 100)
    bar_color = (0, 255, 0) if fill_pct >= 80 else (0, 200, 255) if fill_pct >= 30 else (0, 0, 255)
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)

    # 채움률 텍스트 (대형)
    cv2.putText(display, f"{fill_pct:.0f}%", (w // 2 - 60, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)

    # 상태 판정
    if fill_pct >= 80:
        status_text, status_c = "FULL - Ready to ship", (0, 255, 0)
    elif fill_pct >= 30:
        status_text, status_c = "LOADING...", (0, 200, 255)
    else:
        status_text, status_c = "NEARLY EMPTY", (0, 0, 255)
    cv2.putText(display, status_text, (w // 2 - 120, h // 2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_c, 2)

    cv2.putText(display, f"FILL LEVEL | Avg: {avg_h:.0f}mm",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 180), 2)
    return display


# ============================================================
# 3. MJPEG 스트림 제너레이터
# ============================================================

# 미션 번호 → 처리 함수 매핑 딕셔너리
MISSION_FUNCS = {
    1: process_m1, 2: process_m2, 3: process_m3,
    4: process_m4, 5: process_m5, 6: process_m6, 7: process_m7,
    41: process_m41, 51: process_m51, 61: process_m61,
}

def generate_stream(mission_id):
    """미션별 MJPEG 스트림을 생성하는 제너레이터 함수.

    Flask Response에서 호출되며, 무한 루프로 프레임을 yield한다.
    try-except로 감싸서 어떤 에러에도 generator가 죽지 않게 보호한다.
    """
    # 인터랙션 상태 초기화 (미션 전환 시 이전 상태 클리어)
    state["click_point"] = None
    state["click_depth"] = 0.0
    state["click_3d"] = None
    state["click_pixel"] = None
    state["click_history"] = []
    state["roi_start"] = None
    state["roi_end"] = None
    state["roi_stats"] = None

    func = MISSION_FUNCS.get(mission_id)  # 미션 처리 함수 조회

    # [2026-02-22 수정] active_mission과 일치할 때만 실행 → 미션 전환 시 이전 generator 자동 종료
    while active_mission == mission_id:
        try:
            # [2026-02-22 수정] 공장 미션(41/51/61)은 필터 적용 Depth 사용
            # → 캡처(배경/양품/빈팔레트)도 필터 적용이므로 동일 조건 맞춤
            use_filter = mission_id in (41, 51, 61)
            color, depth_frame, depth_array = get_frames(apply_filter=use_filter)

            if color is None:
                # 카메라 연결 안 됨 → 대기 이미지 생성
                frame = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for RealSense...",
                            (100, CAMERA_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            elif func:
                # 미션 처리 함수 실행
                frame = func(color, depth_frame, depth_array)
            else:
                frame = color             # 함수 없으면 원본 표시

            if frame is None:             # 처리 실패 시 빈 프레임
                frame = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)

            # JPEG 인코딩 (품질 80%)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # MJPEG boundary 형식으로 yield
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        except Exception as e:
            # [2026-02-21 추가] generator 불멸성 보장 — 에러가 나도 스트림 유지
            print(f"⚠️ 미션{mission_id} 프레임 오류: {e}")
            time.sleep(0.1)               # 에러 시 CPU 폭주 방지

        time.sleep(0.05)                  # ~20fps 프레임 레이트 제한


# ============================================================
# 4. Flask 라우트 (웹 API 엔드포인트)
# ============================================================

@app.route('/')
def index():
    """메인 페이지 — index.html을 렌더링한다."""
    return render_template('index.html')


@app.route('/mission/<int:mid>/start', methods=['POST'])
def mission_start(mid):
    """미션 시작 — RealSense 미초기화 시 자동 초기화."""
    global active_mission
    active_mission = mid                  # [2026-02-22 추가] 이전 generator 종료 트리거
    if rs_pipeline is None:               # 카메라 미연결 시
        try:
            init_realsense()              # 초기화 시도
        except Exception as e:
            return jsonify({"error": f"RealSense 초기화 실패: {e}"}), 500
    print(f"🎯 미션 {mid} 시작 (active_mission={active_mission})")
    return jsonify({"status": "ok", "mission": mid})


@app.route('/mission/<int:mid>/stop', methods=['POST'])
def mission_stop(mid):
    """미션 종료 — 이전 generator 종료 + 리소스 상태 로깅."""
    global active_mission
    active_mission = 0                    # [2026-02-22 추가] generator while 루프 종료 트리거
    time.sleep(0.1)                       # generator가 종료될 시간 확보
    print(f"🛑 미션 {mid} 종료 (active_mission=0)")
    return jsonify({"status": "ok"})


@app.route('/mission/<int:mid>/feed')
def mission_feed(mid):
    """MJPEG 실시간 스트림 엔드포인트 — 브라우저 <img src> 용."""
    return Response(
        generate_stream(mid),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/mission/<int:mid>/click', methods=['POST'])
def mission_click(mid):
    """클릭 좌표 수신 → 거리(m) 및 3D 좌표(X,Y,Z) 계산."""
    data = request.get_json()             # JSON 파싱
    x = int(data.get('x', 0))            # 클릭 x 좌표
    y = int(data.get('y', 0))            # 클릭 y 좌표
    result = {"x": x, "y": y}

    # Depth 프레임에서 클릭 지점 거리 읽기
    _, depth_frame, _ = get_frames()
    if depth_frame is not None:
        depth_m = depth_frame.get_distance(x, y)  # 미터 단위 거리
        state["click_point"] = (x, y)
        state["click_depth"] = depth_m
        result["depth_m"] = round(depth_m, 4)
        result["depth_cm"] = round(depth_m * 100, 1)

        # 클릭 이력 관리 (최대 5개)
        state["click_history"].append((x, y, depth_m))
        if len(state["click_history"]) > 5:
            state["click_history"].pop(0)   # 가장 오래된 것 제거

        # 3D 좌표 계산 (미션 3용 — 역투영 공식)
        if rs_intrinsics and depth_m > 0:
            X = (x - rs_intrinsics.ppx) * depth_m / rs_intrinsics.fx  # X = (u-cx)*Z/fx
            Y = (y - rs_intrinsics.ppy) * depth_m / rs_intrinsics.fy  # Y = (v-cy)*Z/fy
            state["click_3d"] = (X, Y, depth_m)   # Z = depth 그대로
            state["click_pixel"] = (x, y)
            result["X_cm"] = round(X * 100, 1)
            result["Y_cm"] = round(Y * 100, 1)
            result["Z_cm"] = round(depth_m * 100, 1)

    return jsonify(result)


@app.route('/mission/<int:mid>/key', methods=['POST'])
def mission_key(mid):
    """키 입력 수신 — 컬러맵 변경 (미션 4)."""
    data = request.get_json()
    key = data.get('key', '')             # 입력된 키
    result = {"key": key}

    # 키 → 컬러맵 매핑 테이블
    cmap_map = {
        '1': ("JET", cv2.COLORMAP_JET),       # 기본 (가까움=빨강)
        '2': ("TURBO", cv2.COLORMAP_TURBO),   # 고대비 무지개
        '3': ("BONE", cv2.COLORMAP_BONE),     # 흑백+파랑 톤
        '4': ("HOT", cv2.COLORMAP_HOT),       # 열화상 스타일
    }
    if key in cmap_map:
        state["colormap_name"], state["colormap"] = cmap_map[key]
        result["colormap"] = state["colormap_name"]

    return jsonify(result)


@app.route('/mission/<int:mid>/roi', methods=['POST'])
def mission_roi(mid):
    """ROI 드래그 이벤트 → 영역 Depth 통계 계산 (미션 5)."""
    data = request.get_json()
    action = data.get('action', '')       # start / end / reset

    if action == 'start':
        # 드래그 시작점 저장
        state["roi_start"] = [int(data['x']), int(data['y'])]
        state["roi_end"] = [int(data['x']), int(data['y'])]
        state["roi_stats"] = None

    elif action == 'end':
        # 드래그 끝점 저장 + 통계 계산
        state["roi_end"] = [int(data['x']), int(data['y'])]
        _, _, depth_array = get_frames()
        if depth_array is not None and state["roi_start"] and state["roi_end"]:
            # 좌상단/우하단 정렬
            x1 = min(state["roi_start"][0], state["roi_end"][0])
            y1 = min(state["roi_start"][1], state["roi_end"][1])
            x2 = max(state["roi_start"][0], state["roi_end"][0])
            y2 = max(state["roi_start"][1], state["roi_end"][1])

            if x2 - x1 > 5 and y2 - y1 > 5:     # 최소 크기 체크
                roi = depth_array[y1:y2, x1:x2]  # ROI 영역 추출
                # 유효 픽셀만 선택 (0 = 측정 불가)
                valid = roi[roi > 0].astype(np.float64) * rs_depth_scale
                if len(valid) > 0:
                    state["roi_stats"] = {
                        "mean": float(np.mean(valid)),      # 평균 거리
                        "min": float(np.min(valid)),        # 최소 거리
                        "max": float(np.max(valid)),        # 최대 거리
                        "std": float(np.std(valid)),        # 표준편차
                        "valid_pct": len(valid) / roi.size * 100,  # 유효 비율
                        "roi": (x1, y1, x2, y2),            # ROI 좌표
                    }

    elif action == 'reset':
        # ROI 초기화
        state["roi_start"] = None
        state["roi_end"] = None
        state["roi_stats"] = None

    return jsonify({"status": "ok", "stats": state["roi_stats"]})


@app.route('/mission/<int:mid>/settings', methods=['POST'])
def mission_settings(mid):
    """슬라이더 설정 수신 (미션 7 — 감지 영역 비율, 임계 거리)."""
    data = request.get_json()
    if 'alert_zone_pct' in data:
        state["alert_zone_pct"] = int(data["alert_zone_pct"])   # ROI 비율 (%)
    if 'alert_dist_m' in data:
        state["alert_dist_m"] = float(data["alert_dist_m"])     # 임계 거리 (m)
    return jsonify({
        "status": "ok",
        "alert_zone_pct": state["alert_zone_pct"],
        "alert_dist_m": state["alert_dist_m"],
    })


# [2026-02-22 추가] 기준 프레임 캡처 API (4-1, 5-1, 6-1 미션용)
@app.route('/mission/<int:mid>/capture', methods=['POST'])
def mission_capture(mid):
    """기준 프레임 캡처 — 5프레임 평균으로 안정적 기준 Depth 생성."""
    frames_list = []                      # Depth 프레임 누적 리스트
    for i in range(5):
        _, _, depth_array = get_frames(apply_filter=True)  # 필터 적용하여 노이즈 최소화
        if depth_array is not None:
            frames_list.append(depth_array.astype(np.float64))
        time.sleep(0.05)                  # 프레임 간격

    if len(frames_list) < 3:              # 최소 3프레임 필요
        return jsonify({"error": "프레임 부족 — 카메라 확인"}), 500

    avg_depth = np.mean(frames_list, axis=0)  # 5프레임 평균

    # 미션별 기준 저장
    if mid == 41:
        ref_frames["bg_depth"] = avg_depth
        label = "배경(빈 컨베이어)"
    elif mid == 51:
        ref_frames["golden_depth"] = avg_depth
        label = "양품(Golden Sample)"
    elif mid == 61:
        ref_frames["empty_depth"] = avg_depth
        label = "빈 팔레트"
    else:
        return jsonify({"error": f"미션 {mid}는 캡처 미지원"}), 400

    center_mm = avg_depth[CAMERA_H // 2, CAMERA_W // 2] * rs_depth_scale * 1000
    print(f"📸 미션 {mid}: {label} 기준 캡처 완료 (중앙 depth: {center_mm:.1f}mm)")

    return jsonify({
        "status": "ok",
        "label": label,
        "center_depth_mm": round(center_mm, 1),
        "frames_used": len(frames_list),
    })


# [2026-02-22 추가] 미션 4-1 규격 설정 API
@app.route('/mission/41/setspec', methods=['POST'])
def mission_setspec():
    """현재 제품의 높이를 측정하여 규격으로 설정한다."""
    if ref_frames["bg_depth"] is None:
        return jsonify({"error": "배경 캡처를 먼저 해주세요"}), 400

    # 5프레임 평균으로 안정적 높이 측정
    heights = []
    for _ in range(5):
        _, _, depth_array = get_frames(apply_filter=True)
        if depth_array is None:
            continue
        h, w = depth_array.shape
        ry1, ry2 = int(h * 0.2), int(h * 0.8)      # 중앙 60% ROI
        rx1, rx2 = int(w * 0.2), int(w * 0.8)
        bg_roi = ref_frames["bg_depth"][ry1:ry2, rx1:rx2]
        now_roi = depth_array[ry1:ry2, rx1:rx2].astype(np.float64)
        diff_mm = (bg_roi - now_roi) * rs_depth_scale * 1000
        diff_mm = np.clip(diff_mm, 0, 2000)
        mask = diff_mm > 30
        if np.sum(mask) > 2000:
            heights.append(float(np.percentile(diff_mm[mask], 95)))
        time.sleep(0.05)

    if len(heights) < 3:
        return jsonify({"error": "제품이 감지되지 않습니다. 제품을 놓고 다시 시도하세요"}), 400

    spec_height = round(float(np.mean(heights)), 1)   # 5프레임 평균 높이
    ref_frames["spec_height_mm"] = spec_height         # 규격으로 등록
    print(f"📏 규격 설정: {spec_height}mm ±{ref_frames['spec_tolerance_mm']}mm")

    return jsonify({
        "status": "ok",
        "spec_height_mm": spec_height,
        "tolerance_mm": ref_frames["spec_tolerance_mm"],
    })


@app.route('/status')
def status_api():
    """서버 상태 확인 API — 연결 진단용."""
    return jsonify({
        "server": "running",
        "realsense": rs_pipeline is not None,
        "depth_scale": rs_depth_scale,
    })


# ============================================================
# 5. 서버 시작
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🏭 RealSense LOCAL — NUC 직결 미션 서버")
    print("=" * 60)
    print(f"🌐 http://localhost:{FLASK_PORT}")
    print("📷 RealSense D435i/D455 USB 직결 전용")
    print("🎯 미션 1~7 (Type A LOCAL)")
    print("=" * 60)

    # Flask 서버 실행 (threaded=True → 동시 요청 처리 허용)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
