# 단계별 재고조사 서버 — 카메라→드래그영역→확대→SAM→(CountGD+rim 비교)→저장
# [2026-06-28 재작성] YOLO 제거, 마우스 드래그로 영역 지정, rim 카운팅 추가

import os
import re
import cv2
import json
import time
import base64
import threading
import datetime

import numpy as np
import requests
from scipy.signal import find_peaks    # rim 줄무늬 peak 검출용
from flask import Flask, render_template, Response, jsonify, request

# ============================================================
# ★ 사용자 환경에 맞게 수정할 설정값들
# ============================================================
CAM_INDEX = 0                       # ← 누크 웹카메라 번호
CAM_WIDTH = 1280                    # ← 카메라 가로 해상도
CAM_HEIGHT = 720                    # ← 카메라 세로 해상도
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000                   # ← 누크 웹서버 포트

GPU_IP = "192.168.0.75"            # ← 4090 PC IP
COUNTGD_URL = f"http://{GPU_IP}:5005/count"  # CountGD 카운팅 서버
SAM3_URL = f"http://{GPU_IP}:5001/segment"   # SAM3 박스 프롬프트 세그멘테이션
GDINO_URL = f"http://{GPU_IP}:5004/detect"   # Grounding DINO 텍스트 탐지 (AI가 컵 찾기)
VLM_URL = "http://192.168.0.36:5002/analyze" # InternVL VLM 카운팅 서버
GDINO_PROMPT = "cup"               # ← GDINO 탐지 프롬프트
GDINO_BOX_THRESHOLD = 0.30         # ← 박스 신뢰도 임계값
GDINO_TEXT_THRESHOLD = 0.25        # ← 텍스트 매칭 임계값
COUNTGD_PROMPT = "cup"             # ← 카운팅 대상 텍스트
INVENTORY_JSON = "inventory.json"  # 재고 기록 저장 파일

# 누크 로컬 YOLO (1차 탐지)
YOLO_MODEL_PATH = "/home/zeta/lerobot/yolo26n_openvino_model/"  # ← 누크 YOLO 경로
YOLO_IMGSZ = 640                   # ← OpenVINO 변환 크기와 일치
YOLO_CUP_CLASS = 41                # ← COCO 'cup' 클래스 id

# ============================================================
# 전역 상태
# ============================================================
app = Flask(__name__)
latest_frame = None                      # 카메라 최신 프레임
latest_frame_lock = threading.Lock()     # 프레임 동기화
yolo_model = None                        # 누크 YOLO 모델

session_state = {
    "captured": None,      # 촬영한 원본 프레임
    "yolo_boxes": [],      # YOLO 1차 탐지 박스
    "gdino_boxes": [],     # GDINO 재확인 박스
    "best_box": None,      # 가장 큰 컵 박스 (스택)
    "masked_crop": None,   # SAM3 분할 후 확대한 컵 (배경 제거)
    "countgd_n": 0,        # CountGD 결과
    "vlm_n": 0,            # VLM 결과
    "rim_n": 0,            # rim 카운팅 결과
    "final": 0             # 최종 채택 개수
}
session_lock = threading.Lock()


# ============================================================
# 1. 카메라 캡처 스레드
# ============================================================
def camera_capture_thread():
    """누크 웹카메라에서 계속 프레임을 읽는 스레드."""
    global latest_frame
    cap = cv2.VideoCapture(CAM_INDEX)                        # 카메라 열기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)            # 가로 해상도
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)         # 세로 해상도
    print(f"📷 카메라 시작: index={CAM_INDEX}")
    while True:
        ok, frame = cap.read()                              # 프레임 읽기
        if ok:
            with latest_frame_lock:
                latest_frame = frame                        # 최신 프레임 갱신
        else:
            time.sleep(0.1)


# ============================================================
# 2. 헬퍼
# ============================================================
def _encode_b64(image_bgr):
    """BGR 이미지를 JPEG base64로 인코딩."""
    _, jpeg = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])  # JPEG 압축
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')                     # base64 변환


def yolo_detect_cups(image_bgr):
    """누크 YOLO로 컵을 1차 탐지한다. 반환: 박스 리스트."""
    if yolo_model is None:
        return []
    results = yolo_model(image_bgr, verbose=False, imgsz=YOLO_IMGSZ, conf=0.25)  # YOLO 추론
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    cls = boxes.cls.cpu().numpy().astype(int)                    # 클래스
    xyxy = boxes.xyxy.cpu().numpy().astype(int)                  # 박스 좌표
    return [list(map(int, xyxy[i])) for i in range(len(cls)) if cls[i] == YOLO_CUP_CLASS]


def box_iou(a, b):
    """두 박스의 IoU(겹침 비율)를 계산한다."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])                 # 교집합 좌상단
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])                 # 교집합 우하단
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih                                            # 교집합 면적
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter  # 합집합
    return inter / ua if ua > 0 else 0.0


def cross_verify(yolo_boxes, gdino_boxes, iou_thresh=0.3):
    """YOLO와 GDINO가 둘 다 컵이라고 한 박스만 신뢰한다 (2단 검증)."""
    if not yolo_boxes:
        return gdino_boxes                                     # YOLO 없으면 GDINO만
    if not gdino_boxes:
        return yolo_boxes                                     # GDINO 없으면 YOLO만
    verified = [g for g in gdino_boxes
                if any(box_iou(g, y) > iou_thresh for y in yolo_boxes)]  # 겹치는 것만
    return verified if verified else gdino_boxes               # 없으면 GDINO 사용


def call_gdino(image_bgr, prompt=GDINO_PROMPT):
    """GDINO 서버에 이미지를 보내 AI가 스스로 컵 박스를 찾게 한다.

    사람이 박스를 그리는 대신 AI가 "cup"이라는 말만 듣고 컵을 찾는다.
    반환: 박스 리스트 [[x1,y1,x2,y2], ...]
    """
    try:
        payload = {
            "image": _encode_b64(image_bgr),                       # 이미지
            "text": prompt,                                        # "cup"
            "box_threshold": GDINO_BOX_THRESHOLD,                  # 박스 임계값
            "text_threshold": GDINO_TEXT_THRESHOLD                 # 텍스트 임계값
        }
        r = requests.post(GDINO_URL, json=payload, timeout=30)     # GDINO 호출
        data = r.json()
        dets = data.get("detections", data.get("boxes", []))       # 탐지 결과
        boxes = []
        for d in dets:
            if isinstance(d, dict):
                b = d.get("box", d.get("bbox"))                    # dict 형식
            else:
                b = d                                              # 리스트 형식
            if b and len(b) == 4:
                boxes.append([int(v) for v in b])                  # 박스 좌표
        return boxes
    except Exception as e:
        print(f"⚠️ GDINO 호출 실패: {e}")
        return []


def call_vlm_count(crop_bgr):
    """InternVL VLM에 이미지를 보내 포개진 컵 개수를 추론하게 한다.

    rim/CountGD가 기계적으로 세는 것과 달리, VLM은 맥락으로 판단한다.
    디자인 줄무늬와 컵 경계를 구분할 수 있어 무늬 있는 컵에 강하다.
    반환: 개수(int) 또는 -1(실패)
    """
    prompt = (
        "이 이미지는 옆에서 본, 포개져 쌓인 종이컵 더미입니다. "
        "컵이 몇 개 쌓여 있는지 세어주세요. "
        "컵의 디자인 무늬(줄무늬, 그림)는 컵 개수와 무관하니 헷갈리지 마세요. "
        "포개진 컵의 테두리(rim) 단 수를 세는 것이 핵심입니다. "
        "답은 반드시 숫자 하나만, 예: 12"
    )
    try:
        payload = {"image": _encode_b64(crop_bgr), "prompt": prompt}  # 이미지+프롬프트
        r = requests.post(VLM_URL, json=payload, timeout=60)         # VLM 호출 (느림)
        data = r.json()
        text = data.get("analysis", data.get("answer", ""))         # 응답 텍스트
        nums = re.findall(r'\d+', text)                             # 숫자 추출
        if nums:
            return int(nums[0])                                    # 첫 숫자 = 개수
        return -1
    except Exception as e:
        print(f"⚠️ VLM 호출 실패: {e}")
        return -1


def call_countgd(crop_bgr, prompt=COUNTGD_PROMPT):
    """확대 crop을 4090 CountGD 서버에 보내 카운팅한다."""
    try:
        payload = {"image": _encode_b64(crop_bgr), "text": prompt}  # 요청 본문
        r = requests.post(COUNTGD_URL, json=payload, timeout=30)    # CountGD 호출
        data = r.json()
        return int(data.get("count", 0))                            # 개수
    except Exception as e:
        print(f"⚠️ CountGD 호출 실패: {e}")
        return -1                                                  # -1 = 실패


def call_sam3_box(image_bgr, box_xyxy, label="cup"):
    """SAM3 서버에 박스를 보내 그 안의 정밀 마스크(컵 영역)를 받는다.

    빅맨님 SAM3 서버 응답: mask_binary(PNG 0/255 base64), mask_h, mask_w.
    반환: 이진 마스크 numpy (H,W) 또는 None
    """
    try:
        h, w = image_bgr.shape[:2]
        payload = {
            "image": _encode_b64(image_bgr),                       # 전체 이미지
            "boxes": [list(map(int, box_xyxy))],                   # 박스 1개
            "labels": [label]                                      # 라벨
        }
        r = requests.post(SAM3_URL, json=payload, timeout=30)      # SAM3 호출
        data = r.json()
        mb64 = data.get("mask_binary")                             # 통합 마스크 PNG base64
        if not mb64:
            return None
        raw = base64.b64decode(mb64)                              # base64 디코딩
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)  # PNG→그레이
        if arr is None:
            return None
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)  # 원본 크기 정렬
        return (arr > 127).astype(np.uint8)                       # 이진 마스크
    except Exception as e:
        print(f"⚠️ SAM3 호출 실패: {e}")
        return None


def apply_mask(image_bgr, mask):
    """마스크 밖(컵이 아닌 부분)을 검게 칠한다. rim 카운팅이 배경에 속지 않게."""
    out = image_bgr.copy()
    out[mask == 0] = 0                                            # 컵 아닌 곳 = 검정
    return out


# ============================================================
# 3. rim 카운팅 (옆에서 본 스택의 줄무늬를 세는 전용 방식)
# ============================================================
def count_rims(crop_bgr, max_cups=30):
    """세로축 밝기 프로파일에서 rim 줄무늬 peak를 세어 컵 개수를 구한다.

    옆에서 포개진 컵은 각 테두리(rim)가 일정 간격의 가로 줄무늬를 만든다.
    세로 방향 밝기 변화의 peak 개수 = 컵 개수.
    반환: (개수, peak y좌표 리스트)
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)             # 그레이 변환
    profile = gray.mean(axis=1)                                  # 각 행의 평균 밝기 (세로 프로파일)
    inv = profile.max() - profile                               # 어두운 rim이 peak가 되도록 반전
    inv = np.convolve(inv, np.ones(3) / 3, mode='same')         # 노이즈 평활

    h = len(inv)
    min_dist = max(5, h // (max_cups * 2))                      # rim 사이 최소 간격
    peaks, _ = find_peaks(inv, distance=min_dist,              # peak 검출
                          prominence=inv.std() * 0.5)
    n = min(len(peaks), max_cups)                              # 개수 (상한 제한)
    return n, peaks.tolist()


def count_rims_masked(masked_bgr, max_cups=30):
    """SAM3로 배경 제거된 crop에서 컵 영역만 분석해 rim을 센다.

    검은 배경(마스크 밖)을 제외하고, 컵 픽셀이 충분한 행만 본다.
    이렇게 하면 배경·옷·그림자가 rim으로 오인되지 않는다.
    """
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)          # 그레이 변환
    # 각 행에서 컵 픽셀(0이 아닌) 비율 계산
    cup_ratio = (gray > 10).mean(axis=1)                        # 행별 컵 픽셀 비율
    valid_rows = cup_ratio > 0.08                               # 컵 8% 이상 (상단 첫 rim 살림)

    # 컵 영역 행만의 밝기 프로파일 (배경 행은 제외)
    profile = np.array([gray[y][gray[y] > 10].mean() if valid_rows[y] else 0
                        for y in range(gray.shape[0])])         # 컵 픽셀 평균만
    if profile.max() == 0:
        return 0, []

    inv = profile.max() - profile                              # rim(어두운 띠)을 peak로
    inv[~valid_rows] = 0                                       # 배경 행은 0으로 눌러 무시
    inv = np.convolve(inv, np.ones(3) / 3, mode='same')        # 평활

    h = len(inv)
    min_dist = max(4, h // (max_cups * 2))                     # 최소 간격 (약간 완화)
    peaks, _ = find_peaks(inv, distance=min_dist,             # peak 검출
                          prominence=inv.std() * 0.4)          # 임계값 낮춤 (약한 rim도)
    if len(peaks) == 0:
        return 0, []

    # ── 간격 규칙성 필터: 멀리 떨어진 가짜 peak 제거 (컵 몸통 그림자 등) ──
    # 아래로 좁아지는 원근 효과는 허용하고 "큰 점프"(3배 초과)만 컷
    peaks = np.array(sorted(peaks))
    if len(peaks) >= 4:
        gaps = np.diff(peaks)                                  # 인접 peak 간격
        median_gap = np.median(gaps)                          # 대표 간격
        kept = [int(peaks[0])]
        for i in range(1, len(peaks)):
            if (peaks[i] - kept[-1]) <= median_gap * 3.0:      # 3배 넘는 큰 점프만 컷
                kept.append(int(peaks[i]))
            else:
                break                                          # 큰 간격 = 몸통, 중단
        peaks = np.array(kept)

    n = min(len(peaks), max_cups)
    return n, peaks.tolist()


def draw_rim_lines(crop_bgr, peaks):
    """검출된 rim 위치에 가로선을 그려 시각화한다."""
    vis = crop_bgr.copy()
    for i, y in enumerate(peaks):
        cv2.line(vis, (0, int(y)), (vis.shape[1], int(y)), (60, 200, 90), 2)  # 초록 가로선
        cv2.putText(vis, str(i + 1), (5, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 90), 2)          # 번호
    return vis


def segment_cup_preview(crop_bgr):
    """확대 crop에서 컵 경계를 시각화 (교육용 SAM 미리보기)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)            # 그레이
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu 이진화
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)        # 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)       # 구멍 메우기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = crop_bgr.copy()
    if contours:
        c = max(contours, key=cv2.contourArea)                  # 최대 윤곽
        colored = crop_bgr.copy()
        cv2.drawContours(colored, [c], -1, (237, 58, 124), thickness=cv2.FILLED)
        overlay = cv2.addWeighted(crop_bgr, 0.6, colored, 0.4, 0)  # 반투명 합성
        cv2.drawContours(overlay, [c], -1, (237, 58, 124), 3)   # 경계선
    return overlay


# ============================================================
# 4. 라이브 스트림
# ============================================================
def generate_live_stream():
    """카메라 라이브 영상 MJPEG 스트림 (YOLO 없음)."""
    while True:
        with latest_frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)  # 대기 화면
            cv2.putText(frame, "Waiting camera...", (80, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.06)


# ============================================================
# 5. 단계별 라우트
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_live_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── 1단계: 촬영 + YOLO 1차 탐지 ──
@app.route('/step1_yolo', methods=['POST'])
def step1_yolo():
    """현재 프레임을 촬영하고, YOLO로 컵을 1차 탐지한다."""
    with latest_frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "카메라 프레임 없음"}), 503

    yolo_boxes = yolo_detect_cups(frame)                         # YOLO 1차 탐지
    if not yolo_boxes:
        return jsonify({"error": "YOLO가 컵을 찾지 못했습니다. 컵이 잘 보이게 한 뒤 다시 촬영하세요."}), 422

    with session_lock:
        session_state["captured"] = frame
        session_state["yolo_boxes"] = yolo_boxes

    preview = frame.copy()
    for i, b in enumerate(yolo_boxes):
        cv2.rectangle(preview, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)   # 초록(YOLO)
        cv2.putText(preview, f"cup {i+1}", (b[0], b[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return jsonify({"ok": True, "image": _encode_b64(preview), "count": len(yolo_boxes),
                    "message": f"YOLO가 컵 {len(yolo_boxes)}곳을 1차 탐지했습니다."})


# ── 2단계: GDINO 재확인 (2단 검증) ──
@app.route('/step2_gdino', methods=['POST'])
def step2_gdino():
    """GDINO로 컵을 재탐지하고, YOLO 결과와 교차검증한다."""
    with session_lock:
        frame = session_state["captured"]
        yolo_boxes = session_state["yolo_boxes"]
    if frame is None:
        return jsonify({"error": "먼저 YOLO 탐지를 하세요 (1단계)"}), 400

    gdino_boxes = call_gdino(frame, GDINO_PROMPT)                # GDINO 재탐지
    verified = cross_verify(yolo_boxes, gdino_boxes)            # 2단 검증

    if not verified:
        return jsonify({"error": "GDINO 재확인에서 컵이 검증되지 않았습니다."}), 422

    # 가장 큰 박스 = 스택 (SAM3 대상)
    best = max(verified, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))  # 최대 면적

    with session_lock:
        session_state["gdino_boxes"] = gdino_boxes
        session_state["best_box"] = best

    # 시각화: YOLO(초록) + GDINO검증(파랑) + 선택된 스택(빨강)
    preview = frame.copy()
    for b in gdino_boxes:
        cv2.rectangle(preview, (b[0], b[1]), (b[2], b[3]), (255, 150, 0), 2)  # 파랑(GDINO)
    cv2.rectangle(preview, (best[0], best[1]), (best[2], best[3]), (0, 80, 230), 4)  # 빨강(선택)
    cv2.putText(preview, "SELECTED STACK", (best[0], best[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 230), 2)

    return jsonify({"ok": True, "image": _encode_b64(preview),
                    "yolo": len(yolo_boxes), "gdino": len(gdino_boxes), "verified": len(verified),
                    "message": f"2단 검증 완료. YOLO {len(yolo_boxes)} + GDINO {len(gdino_boxes)} → 가장 큰 컵 1개 선택."})


# ── 3단계: SAM3 분할 (원본 크기 그대로, 배경만 제거) ──
@app.route('/step3_sam', methods=['POST'])
def step3_sam():
    """선택된 컵을 SAM3로 분할한다. 자르기·확대 없이 원본 크기에서 배경만 제거.

    빅맨님 흐름: SAM3 분할 상태만 그대로 보여주고, 그대로 카운팅.
    """
    with session_lock:
        frame = session_state["captured"]
        best = session_state["best_box"]
    if frame is None or best is None:
        return jsonify({"error": "먼저 GDINO 재확인을 하세요 (2단계)"}), 400

    # SAM3에 선택된 박스를 줘서 그 안의 컵을 정밀 분할
    mask = call_sam3_box(frame, best, label="cup")              # SAM3 마스크 요청 (원본 크기)

    H, W = frame.shape[:2]
    if mask is None or mask.sum() < 100:
        # SAM3 실패 → 박스 안만 살리는 폴백
        method = "박스 폴백 (SAM3 실패)"
        mask = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = best
        mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = 1     # 박스 영역만 1
    else:
        method = "SAM3"

    # 원본 프레임 그대로, 컵 밖만 검게 (자르기·확대 없음)
    masked_full = frame.copy()
    masked_full[mask == 0] = 0                                  # 컵 아닌 곳 = 검정

    with session_lock:
        session_state["masked_crop"] = masked_full             # 원본 크기 분할 결과

    # 시각화: 컵 경계선(보라색)으로 분할 상태 표시
    vis = masked_full.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (237, 58, 124), 2)      # 컵 경계선
    cup_px = int(mask.sum())                                   # 컵 픽셀 수

    return jsonify({"ok": True, "image": _encode_b64(vis),
                    "message": f"{method}로 컵을 분리했습니다 (컵 영역 {cup_px:,}px). 원본 크기 그대로 카운팅합니다."})


# ── 4단계: 카운팅 비교 (CountGD vs rim) ──
@app.route('/step4_count', methods=['POST'])
def step4_count():
    """SAM3로 배경 제거한 영역을 rim·CountGD·VLM 세 방식으로 세어 비교한다."""
    with session_lock:
        target = session_state["masked_crop"]                # SAM3 분리된 컵
    if target is None:
        return jsonify({"error": "먼저 SAM3 분할을 하세요 (3단계)"}), 400

    # [방식 1] rim 카운팅 (누크, 줄무늬 신호) — 민무늬 컵에 강함
    rim_n, peaks = count_rims_masked(target)                # rim 카운팅
    rim_vis = draw_rim_lines(target, peaks)                 # 검출선 시각화

    # [방식 2] CountGD (4090, 객체 탐지)
    countgd_n = call_countgd(target)                        # CountGD 카운팅
    countgd_ok = countgd_n >= 0

    # [방식 3] VLM (InternVL, 맥락 추론) — 무늬 있는 컵에 강함
    vlm_n = call_vlm_count(target)                          # VLM 카운팅
    vlm_ok = vlm_n >= 0

    final = rim_n                                           # 기본 채택 (사용자가 수정 가능)

    with session_lock:
        session_state["countgd_n"] = countgd_n if countgd_ok else 0
        session_state["vlm_n"] = vlm_n if vlm_ok else 0
        session_state["rim_n"] = rim_n
        session_state["final"] = final

    msg = f"rim: {rim_n}개"
    msg += f" | CountGD: {countgd_n}개" if countgd_ok else " | CountGD: 실패"
    msg += f" | VLM: {vlm_n}개" if vlm_ok else " | VLM: 실패"

    return jsonify({
        "ok": True,
        "rim": rim_n,
        "countgd": countgd_n if countgd_ok else None,
        "vlm": vlm_n if vlm_ok else None,
        "final": final,
        "rim_image": _encode_b64(rim_vis),
        "message": msg
    })


# ── 5단계: 저장 ──
@app.route('/step5_save', methods=['POST'])
def step5_save():
    """최종 결과를 서버 JSON에 저장한다."""
    data = request.get_json() or {}
    with session_lock:
        final = session_state["final"]
        countgd_n = session_state["countgd_n"]
        vlm_n = session_state["vlm_n"]
        rim_n = session_state["rim_n"]

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "location": data.get("location", "미지정"),
        "total": data.get("final", final),                   # 사용자가 수정 가능
        "rim_count": rim_n,                                  # rim 결과
        "countgd_count": countgd_n,                          # CountGD 결과
        "vlm_count": vlm_n                                   # VLM 결과
    }

    records = []
    if os.path.exists(INVENTORY_JSON):
        try:
            with open(INVENTORY_JSON, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
    records.append(record)
    with open(INVENTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return jsonify({"ok": True, "record_count": len(records),
                    "message": f"저장 완료! (총 {len(records)}건)"})


@app.route('/history')
def history():
    if not os.path.exists(INVENTORY_JSON):
        return jsonify({"records": []})
    with open(INVENTORY_JSON, "r", encoding="utf-8") as f:
        return jsonify({"records": json.load(f)})


@app.route('/report')
def report():
    """저장된 재고조사 기록만 보여주는 보고서 페이지."""
    records = []
    if os.path.exists(INVENTORY_JSON):
        try:
            with open(INVENTORY_JSON, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
    # 최신 기록이 위로 오도록 역순 정렬
    records = list(reversed(records))
    total_sum = sum(int(r.get("total", 0)) for r in records)     # 전체 컵 합계
    return render_template('report.html', records=records,
                           total_records=len(records), total_sum=total_sum)


@app.route('/status')
def status():
    with latest_frame_lock:
        has_frame = latest_frame is not None
    return jsonify({"camera_connected": has_frame, "gpu_ip": GPU_IP})


# ============================================================
# 6. 서버 시작
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("📦 단계별 컵 재고조사 — YOLO→GDINO→SAM3→확대→(CountGD+rim 비교)")
    print("   [v2026-06-28 YOLO+GDINO 2단검증판]")
    print("=" * 60)

    # 누크 YOLO 로드 (1차 탐지)
    try:
        from ultralytics import YOLO                         # YOLO 임포트
        print(f"⚡ YOLO 로딩: {YOLO_MODEL_PATH}")
        yolo_model = YOLO(YOLO_MODEL_PATH, task="detect")    # 모델 로드
        print("✅ YOLO 준비 완료")
    except Exception as e:
        print(f"⚠️ YOLO 로드 실패 — 1단계는 GDINO만으로 동작: {e}")
        yolo_model = None

    cam_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    cam_thread.start()

    print(f"🌐 웹 서버: http://{FLASK_HOST}:{FLASK_PORT}")
    print("=" * 60)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
