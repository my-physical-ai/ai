# app.py
# Flask 웹서버 — 생수병 3D 정밀 검사 (룰 기반 depth 비교)
# 환경: NUC에 RealSense D435i 직접 연결
# UI 흐름: ① 양품 5장 등록(평균) → ② 현재 3D 촬영 → ③ 비교 → 결과

import os
import time
import base64

import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify

from realsense_camera import RealSenseCamera   # RealSense 캡처 모듈
from dent_detector import DentDetector         # 찌그러짐 검출 로직

# ============================================================
# ★ 사용자 환경에 맞게 수정할 설정값
# ============================================================
FLASK_HOST = "0.0.0.0"          # ← 모든 네트워크에서 접속 허용
FLASK_PORT = 5000               # ← 웹서버 포트
GOLDEN_PATH = "golden/bottle_golden.npy"  # ← 양품 기준 depth 저장 경로
# ★ 임계값은 카메라-물병 거리에 맞춰 조정 (30~40cm → 5mm, 50cm → 6~7mm 권장)
DENT_THR_MM = 5.0               # ← 찌그러짐 판정 임계 깊이 (mm), 30cm 거리 기준
MIN_AREA_PX = 80                # ← 찌그러짐 최소 면적 (픽셀)
GOLDEN_FRAMES = 5               # ← 양품 등록에 사용할 총 촬영 장수 (연속/한장 공통)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 있으면 사용

# ============================================================
# 전역 객체
# ============================================================
app = Flask(__name__)
camera = RealSenseCamera(width=848, height=480, fps=30)  # 카메라 객체 (848=D435i 최적 해상도)
detector = None                 # 검출기 (카메라 시작 후 생성)

# 양품 등록용 누적 버퍼 (5장을 한 장씩 모음)
golden_buffer = []              # 양품 depth 프레임 누적 리스트
golden_color = None             # 양품 평균 시점의 컬러 이미지 (표시용)

# 현재 촬영(검사 대상) 보관
current_depth = None            # 검사 대상 depth (정지 캡처)
current_color = None            # 검사 대상 컬러 (정지 캡처)


def init_system():
    """카메라를 켜고 검출기를 초기화한다."""
    global detector
    camera.start()              # RealSense 시작 (depth_scale 확정)
    # 검출기 생성 (카메라의 실제 depth_scale 전달)
    detector = DentDetector(
        golden_path=GOLDEN_PATH,
        depth_scale=camera.depth_scale,
        dent_thr_mm=DENT_THR_MM,
        min_area_px=MIN_AREA_PX,
        device=DEVICE,
    )
    print(f"⚙️ 검출기 준비 완료 | device={DEVICE}")


def encode_jpeg_b64(img):
    """이미지를 JPEG base64 문자열로 변환한다 (UI 표시용)."""
    _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])  # JPEG 인코딩
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')             # base64 변환


def depth_to_gray(depth_array):
    """depth 배열을 회색조 시각화 이미지로 변환한다 (가까울수록 밝게)."""
    valid = depth_array > 0                                  # 유효 픽셀
    if valid.sum() == 0:
        return np.zeros((depth_array.shape[0], depth_array.shape[1], 3), np.uint8)
    d = depth_array.astype(np.float32)                       # float 변환
    # 유효 범위만 0~255로 정규화 (반전: 가까울수록 밝게)
    vmin, vmax = d[valid].min(), d[valid].max()
    norm = np.zeros_like(d)
    norm[valid] = 255 * (1 - (d[valid] - vmin) / (vmax - vmin + 1e-6))
    gray = norm.astype(np.uint8)                             # uint8 변환
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)            # 3채널로


# ============================================================
# 1. 실시간 라이브 프리뷰 (촬영 전 화면 확인용)
# ============================================================
def generate_live():
    """현재 카메라 라이브 영상을 MJPEG로 스트리밍 (촬영 위치 확인용)."""
    while True:
        color, _ = camera.get_frames()           # 최신 컬러
        if color is None:
            frame = np.zeros((480, 848, 3), dtype=np.uint8)  # 대기 화면
            cv2.putText(frame, "Waiting for camera...", (250, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        else:
            frame = color
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # 인코딩
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)                         # 약 20fps


# ============================================================
# 2. Flask 라우트
# ============================================================
@app.route('/')
def index():
    """메인 페이지 렌더링."""
    return render_template('index.html')


@app.route('/live_feed')
def live_feed():
    """라이브 프리뷰 MJPEG 스트림 (촬영 위치 확인)."""
    return Response(generate_live(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/golden/reset', methods=['POST'])
def golden_reset():
    """양품 등록 버퍼를 초기화한다 (등록 새로 시작)."""
    global golden_buffer, golden_color
    golden_buffer = []                           # 누적 버퍼 비우기
    golden_color = None                          # 컬러도 초기화
    return jsonify({"ok": True, "count": 0, "total": GOLDEN_FRAMES})


@app.route('/golden/shot', methods=['POST'])
def golden_shot():
    """양품을 한 장 촬영해 버퍼에 누적한다 (하나씩 등록)."""
    global golden_buffer, golden_color
    color, depth = camera.get_frames()           # 현재 프레임
    if depth is None:
        return jsonify({"ok": False, "msg": "카메라 프레임 없음"}), 503

    golden_buffer.append(depth.astype(np.float32))  # depth 누적
    golden_color = color                            # 최신 컬러 보관
    count = len(golden_buffer)                       # 현재 장수

    return jsonify({
        "ok": True,
        "count": count,                          # 촬영된 장수
        "total": GOLDEN_FRAMES,                  # 목표 장수
        "done": count >= GOLDEN_FRAMES,          # 완료 여부
    })


@app.route('/golden/burst', methods=['POST'])
def golden_burst():
    """양품을 연속으로 5장 자동 촬영해 등록한다 (연속촬영)."""
    global golden_buffer, golden_color
    golden_buffer = []                           # 버퍼 초기화
    for _ in range(GOLDEN_FRAMES):
        color, depth = camera.get_frames()       # 프레임 취득
        if depth is not None:
            golden_buffer.append(depth.astype(np.float32))  # 누적
            golden_color = color                            # 컬러 보관
        time.sleep(0.05)                         # 프레임 간격

    if len(golden_buffer) == 0:
        return jsonify({"ok": False, "msg": "카메라 프레임 없음"}), 503

    return _finalize_golden()                    # 평균내어 확정


@app.route('/golden/finalize', methods=['POST'])
def golden_finalize():
    """누적된 양품 프레임들을 평균내어 기준으로 확정·저장한다."""
    if len(golden_buffer) == 0:
        return jsonify({"ok": False, "msg": "촬영된 양품이 없습니다"}), 400
    return _finalize_golden()


def _finalize_golden():
    """양품 버퍼를 평균내어 검출기에 등록하고 평균 이미지를 반환한다."""
    global golden_color
    # 누적 프레임 평균 (노이즈 감소)
    golden_avg = np.mean(golden_buffer, axis=0).astype(np.uint16)  # 평균 depth
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)       # 폴더 보장
    detector.register_golden(golden_avg, GOLDEN_PATH)             # 검출기에 등록

    # 평균 depth의 회색조 시각화 (UI 표시용)
    gray_vis = depth_to_gray(golden_avg)                         # depth 시각화
    # 컬러가 있으면 컬러를, 없으면 회색조를 표시
    show_img = golden_color if golden_color is not None else gray_vis
    img_b64 = encode_jpeg_b64(show_img)                          # base64 변환
    depth_b64 = encode_jpeg_b64(gray_vis)                        # depth 시각화

    print(f"📌 양품 등록 완료 ({len(golden_buffer)}장 평균)")
    return jsonify({
        "ok": True,
        "msg": f"양품 등록 완료 ({len(golden_buffer)}장 평균)",
        "count": len(golden_buffer),
        "image": img_b64,                        # 양품 컬러 이미지
        "depth": depth_b64,                      # 양품 depth 시각화
    })


@app.route('/capture_current', methods=['POST'])
def capture_current():
    """검사 대상(현재)을 정지 촬영해 보관·표시한다."""
    global current_depth, current_color
    color, depth = camera.get_frames()           # 현재 프레임
    if color is None or depth is None:
        return jsonify({"ok": False, "msg": "카메라 프레임 없음"}), 503

    current_depth = depth.copy()                 # 검사 대상 depth 보관
    current_color = color.copy()                 # 검사 대상 컬러 보관

    img_b64 = encode_jpeg_b64(current_color)     # 컬러 base64
    depth_b64 = encode_jpeg_b64(depth_to_gray(current_depth))  # depth 시각화

    return jsonify({
        "ok": True,
        "msg": "현재 촬영 완료",
        "image": img_b64,                        # 현재 컬러
        "depth": depth_b64,                      # 현재 depth 시각화
    })


@app.route('/compare', methods=['POST'])
def compare():
    """등록된 양품과 현재 촬영을 비교해 찌그러짐을 검출한다."""
    # 양품·현재가 모두 준비됐는지 확인
    if detector.golden_tensor is None:
        return jsonify({"ok": False, "msg": "양품을 먼저 등록하세요"}), 400
    if current_depth is None:
        return jsonify({"ok": False, "msg": "검사 대상을 먼저 촬영하세요"}), 400

    result = detector.detect(current_depth)      # 찌그러짐 검출
    if not result["ok"]:
        return jsonify(result), 400

    # 결과 이미지 (현재 컬러 + 찌그러짐 박스)
    annotated = current_color.copy()
    for b in result["boxes"]:
        cv2.rectangle(annotated, (b["x"], b["y"]),
                      (b["x"]+b["w"], b["y"]+b["h"]), (255, 200, 0), 2)  # 청록 박스
        cv2.putText(annotated, f"{b['depth_mm']}mm", (b["x"], b["y"]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
    v = result["verdict"]                         # 판정
    vc = (0, 0, 255) if v == "BLOCK" else (0, 200, 0)  # 색상
    cv2.putText(annotated, v, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, vc, 3)

    annotated_b64 = encode_jpeg_b64(annotated)               # 결과 이미지
    heatmap_b64 = encode_jpeg_b64(detector.make_heatmap(result["diff_map"]))  # 히트맵

    # 콘솔 한글 로그
    print(f"🔎 판정={result['verdict']} | 최대깊이={result['max_depth_mm']}mm | "
          f"찌그러짐={len(result['boxes'])}곳 | 유효비율={result['valid_ratio']}")

    return jsonify({
        "ok": True,
        "verdict": result["verdict"],            # PASS / BLOCK
        "max_depth_mm": result["max_depth_mm"],  # 최대 깊이
        "dent_area_px": result["dent_area_px"],  # 면적
        "boxes": result["boxes"],                # 찌그러짐 위치·깊이
        "annotated": annotated_b64,              # 결과 이미지
        "heatmap": heatmap_b64,                  # 편차 히트맵
        "valid_ratio": result["valid_ratio"],    # 유효 비율
        "low_quality": result["low_quality"],    # 품질 경고
    })


@app.route('/status')
def status():
    """서버·카메라·등록 상태 확인."""
    color, _ = camera.get_frames()
    return jsonify({
        "camera": color is not None,                          # 카메라 연결
        "golden_ready": detector.golden_tensor is not None,   # 양품 등록 여부
        "current_ready": current_depth is not None,           # 현재 촬영 여부
        "golden_count": len(golden_buffer),                   # 양품 누적 장수
        "golden_total": GOLDEN_FRAMES,                        # 목표 장수
        "device": DEVICE,                                     # 연산 장치
    })


# ============================================================
# 3. 서버 시작
# ============================================================
if __name__ == '__main__':
    print("=" * 56)
    print("🍶 생수병 3D 정밀 검사 — NUC + D435i (룰 기반 depth 비교)")
    print("=" * 56)
    init_system()               # 카메라 + 검출기 초기화
    print(f"🌐 웹 접속: http://<NUC_IP>:{FLASK_PORT}")
    print("=" * 56)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
