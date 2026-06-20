# Flask 웹서버 — Pi 듀얼 카메라 수신 + YOLO BoT-SORT 트래킹 + Grounding DINO 언어 탐지
# [2026-02-19 추가] Stage 2: 마법사 UI + Flask + ZeroMQ + YOLO + Grounding DINO
# [2026-02-21 수정] 듀얼 카메라(FRONT+TOP) + BoT-SORT/ByteTrack 트래커 전환 추가
# [2026-06-20 추가] GDINO 박스 → 4090 SAM3 서버(HTTP POST) → 정밀 마스크 회신 연동
#
# ★ 3대 머신 역할 분담:
#   Pi 5  (ARM):  카메라 캡처 + JPEG 압축 + ZeroMQ 전송 (가벼운 작업)
#   NUC   (Intel CPU): YOLO 트래킹 + GDINO 박스 탐지 "어디에 있는지(박스)"
#   4090  (GPU): GDINO 박스를 prompt로 받아 "정확한 픽셀 경계(마스크)" 생성
#
# ★ SAM3 흐름: 사용자가 GDINO로 탐지 → UI의 "SAM3 정밀분할" 버튼 클릭
#              → NUC가 마지막 박스+프레임을 4090으로 POST → 마스크 오버레이 회신
#
# ★ 왜 NUC에서 실행하는가?
#   Pi 5 (ARM):  OpenVINO 사용 불가, PyTorch YOLO ≈ 500ms (2fps)
#   NUC i7 (Intel): OpenVINO YOLO ≈ 35ms (28fps) + BoT-SORT CMC 가능
#
# 실행:
#   Pi 터미널1: python send_camera_front.py  (포트 5556)
#   Pi 터미널2: python send_camera_top.py    (포트 5557)
#   NUC:       conda activate lerobot && python app.py
#   브라우저:   http://NUC_IP:5000

import io
import os
import time
import base64
import threading
from collections import defaultdict

import cv2
import zmq
import numpy as np
import torch
import requests                         # [2026-06-20 추가] 4090 SAM3 서버 HTTP 호출용
from PIL import Image
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ============================================================
# ★ 사용자 환경에 맞게 수정할 설정값들
# ============================================================
PI_IP = "192.168.50.111"           # Pi5 실제 IP (hostname -I 로 확인)
FRONT_PORT = 5556                   # FRONT 카메라 ZeroMQ 포트
TOP_PORT = 5557                     # TOP 카메라 ZeroMQ 포트
FLASK_HOST = "0.0.0.0"             # 모든 네트워크에서 접속 허용
FLASK_PORT = 5000                   # Flask 웹서버 포트
YOLO_MODEL_PATH = "/home/zeta/lerobot/yolo26n_openvino_model/"  # OpenVINO 절대경로
YOLO_IMGSZ = 640                    # OpenVINO 변환 크기와 반드시 일치!
CONFIDENCE = 0.3                    # 트래킹 신뢰도 (낮은 값도 2차 매칭에 활용)
TRAIL_LEN = 30                      # 궤적 길이 (프레임 수)
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_BOX_THRESHOLD = 0.35
GDINO_TEXT_THRESHOLD = 0.25

# [2026-06-20 추가] 4090 SAM3 서버 설정
SAM3_SERVER_URL = "http://192.168.0.75:5001"     # ← 4090 PC 실제 IP (현재 0.75)
SAM3_TIMEOUT = 30.0                 # SAM3 추론 대기 최대 시간(초). 첫 호출은 모델 워밍업으로 느림
SAM3_ENABLED = True                 # False면 SAM3 기능 비활성화 (4090 없을 때)

# [2026-02-21 추가] BoT-SORT 커스텀 설정 파일 (스크립트와 같은 폴더)
BOTSORT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort_lekiwi.yaml")
BYTETRACK_YAML = "bytetrack.yaml"   # ultralytics 내장 YAML

# ============================================================
# 전역 상태 (스레드 간 공유)
# ============================================================
app = Flask(__name__)

# [2026-02-21 수정] 듀얼 카메라 프레임 (FRONT + TOP)
frame_front = None                  # FRONT 카메라 최신 프레임
frame_top = None                    # TOP 카메라 최신 프레임
frame_lock = threading.Lock()       # 프레임 접근 동기화

# [2026-02-21 추가] 카메라 뷰 / 트래커 상태
current_camera = "front"            # "front" / "top" / "dual"
current_tracker_name = "botsort"    # "botsort" / "bytetrack"
current_tracker_yaml = BOTSORT_YAML

# 모델 (전역)
yolo_model = None
gdino_processor = None
gdino_model = None

# [2026-02-21 추가] 트래킹 상태 (MJPEG generator에서 사용)
track_lock = threading.Lock()       # model.track() 동기화 (persist 상태 보호)
trails = defaultdict(list)          # 궤적 {tid: [(x,y), ...]}
all_ids = set()                     # 전체 고유 ID
gdino_lock = threading.Lock()       # GDINO 추론 동기화

# [2026-06-20 추가] 마지막 GDINO 탐지 결과 (SAM3 버튼 클릭 시 재사용)
last_gdino_frame = None             # GDINO가 탐지한 원본 프레임 (BGR)
last_gdino_boxes = []               # GDINO 박스 리스트 [[x1,y1,x2,y2], ...]
last_gdino_prompt = ""             # 그때 사용한 텍스트 프롬프트
last_gdino_lock = threading.Lock()  # 위 3개 접근 동기화

# COCO 클래스 이름
COCO = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog",
    24: "backpack", 39: "bottle", 56: "chair", 62: "tv",
    63: "laptop", 67: "phone", 73: "book",
}


# ============================================================
# 0. BoT-SORT 설정 파일 자동 생성
# ============================================================
def create_botsort_yaml(path: str) -> None:
    """LeKiwi 최적화된 BoT-SORT 설정 파일을 생성한다."""
    yaml_content = """# BoT-SORT: LeKiwi 이동 로봇 최적화 설정
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8
fuse_score: true
gmc_method: orb
with_reid: false
proximity_thresh: 0.5
appearance_thresh: 0.25
"""
    with open(path, 'w') as f:
        f.write(yaml_content)
    print(f"[서버] BoT-SORT 설정 생성: {path}")


# ============================================================
# 1. ZeroMQ 프레임 수신 스레드 (듀얼 카메라)
# ============================================================
def zmq_receiver_thread(port: int, camera_name: str) -> None:
    """백그라운드에서 Pi 카메라 프레임을 계속 수신하는 스레드.

    Args:
        port: ZeroMQ 포트 번호
        camera_name: "front" 또는 "top"
    """
    global frame_front, frame_top

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)              # 최신 프레임만 유지 (딜레이 방지)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 3000)            # 수신 타임아웃 3초
    sock.connect(f"tcp://{PI_IP}:{port}")
    print(f"📡 [{camera_name.upper()}] 수신 시작: tcp://{PI_IP}:{port}")

    while True:
        try:
            buf = sock.recv()                                       # JPEG 바이트 수신
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), 1)   # BGR 디코딩
            if frame is not None:
                with frame_lock:
                    if camera_name == "front":
                        frame_front = frame                         # FRONT 프레임 갱신
                    else:
                        frame_top = frame                           # TOP 프레임 갱신
        except zmq.Again:
            pass                                                    # 타임아웃 — 전송 없음
        except Exception as e:
            print(f"❌ [{camera_name.upper()}] 수신 오류: {e}")
            time.sleep(1)


# ============================================================
# 2. 모델 로딩
# ============================================================
def load_models() -> None:
    """YOLO와 Grounding DINO 모델을 로드한다."""
    global yolo_model, gdino_processor, gdino_model

    # YOLO26 OpenVINO 모델 로드
    print(f"⚡ YOLO 모델 로딩: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH, task="detect")
    print("✅ YOLO 모델 준비 완료!")

    # Grounding DINO Tiny 모델 로드
    print(f"🗣️ Grounding DINO 모델 로딩: {GDINO_MODEL_ID}")
    t0 = time.time()
    gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID)
    gdino_model = gdino_model.to("cpu")             # NUC CPU 전용
    gdino_model.eval()                               # 추론 모드
    print(f"✅ Grounding DINO 준비 완료! ({time.time() - t0:.1f}초)")


# ============================================================
# 2-1. [2026-06-20 추가] 4090 SAM3 서버 호출
# ============================================================
def request_sam3_masks(frame_bgr: np.ndarray, boxes_xyxy: list, prompt: str = "") -> dict | None:
    """GDINO 박스+프롬프트를 4090 SAM3 서버로 보내 정밀 마스크를 받아온다.

    NUC는 박스(어디에)만 찾고, 픽셀 단위 정밀 경계는 4090 SAM3가 담당한다.
    이미지(base64)+박스를 HTTP POST로 보내고, 마스크 오버레이 이미지를 회신받는다.

    Args:
        frame_bgr: 원본 BGR 프레임 (GDINO가 탐지한 그 프레임)
        boxes_xyxy: GDINO 박스 리스트 [[x1,y1,x2,y2], ...] (정수 픽셀 좌표)

    Returns:
        성공: SAM3 응답 dict
        실패: {"_error": "원인 메시지"} dict (화면에 표시용)
    """
    # 프레임을 JPEG → base64 인코딩 (네트워크 전송용)
    _, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

    # 4090으로 보낼 JSON 페이로드
    payload = {
        "image": img_b64,                                   # base64 JPEG 원본 프레임
        "boxes": [list(map(int, b)) for b in boxes_xyxy],   # 박스 좌표 (정수)
        "prompt": prompt,                                   # GDINO 텍스트 프롬프트(공식 sam3용)
    }

    try:
        # SAM3 추론 시간 고려해 timeout 넉넉히
        resp = requests.post(f"{SAM3_SERVER_URL}/segment", json=payload, timeout=SAM3_TIMEOUT)
        if resp.status_code != 200:
            # 4090이 4xx/5xx로 응답 — 본문에 담긴 에러 메시지를 그대로 전달
            try:
                body = resp.json()
                detail = body.get("error") or body.get("_error") or str(body)
            except Exception:
                detail = resp.text[:300]                    # JSON 아니면 앞부분만
            print(f"❌ SAM3 서버 {resp.status_code}: {detail}")
            return {"_error": f"4090 오류 {resp.status_code}: {detail}"}
        # 200이지만 본문에 error 키가 있으면(물체 못 찾음 등) 그대로 안내로 전달
        body = resp.json()
        if isinstance(body, dict) and "error" in body:
            return {"_error": body["error"]}
        return body                                         # 정상 결과
    except requests.exceptions.Timeout:
        msg = f"SAM3 추론이 {SAM3_TIMEOUT:.0f}초를 초과했습니다 (첫 호출 워밍업이면 다시 시도)"
        print(f"⏱️ {msg}")
        return {"_error": msg}
    except requests.exceptions.ConnectionError:
        msg = "4090에 연결할 수 없습니다 (IP/포트/방화벽 확인)"
        print(f"❌ {msg}")
        return {"_error": msg}
    except requests.exceptions.RequestException as e:
        print(f"❌ SAM3 서버 호출 실패: {e}")
        return {"_error": f"요청 실패: {e}"}


# ============================================================
# 3. 트래킹 시각화 유틸리티
# ============================================================
def color_for(tid: int) -> tuple:
    """트랙 ID별 고유 BGR 색상."""
    np.random.seed(tid * 7 + 13)
    return tuple(np.random.randint(80, 255, 3).tolist())


def draw_trail(frame: np.ndarray, pts: list, color: tuple) -> None:
    """궤적을 점점 굵은 선으로 그린다."""
    for i in range(1, len(pts)):
        th = int(np.sqrt(i / 2.0) * 2) + 1
        cv2.line(frame, pts[i - 1], pts[i], color, th)


def draw_tracking_results(frame: np.ndarray, results) -> int:
    """model.track() 결과를 프레임에 바운딩박스+궤적으로 시각화한다.

    Args:
        frame: 그릴 프레임 (in-place 수정)
        results: model.track() 결과

    Returns:
        현재 프레임 활성 객체 수
    """
    active = 0
    if results[0].boxes.id is None:
        return active

    boxes = results[0].boxes
    ids = boxes.id.int().cpu().tolist()
    xyxys = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.int().cpu().tolist()
    confs = boxes.conf.cpu().tolist()

    for tid, xyxy, cls, conf in zip(ids, xyxys, clss, confs):
        active += 1
        all_ids.add(tid)
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        c = color_for(tid)

        # 궤적
        trails[tid].append((cx, cy))
        trails[tid] = trails[tid][-TRAIL_LEN:]
        if len(trails[tid]) > 1:
            draw_trail(frame, trails[tid], c)

        # 바운딩 박스 + 라벨
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        nm = COCO.get(cls, f"cls:{cls}")
        lb = f"ID:{tid} {nm} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(lb, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, lb, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame, (cx, cy), 3, c, -1)

    return active


def make_placeholder(text: str = "No signal") -> np.ndarray:
    """카메라 미연결 시 대기 이미지."""
    ph = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(ph, text, (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    return ph


# ============================================================
# 4. YOLO + BoT-SORT 실시간 스트리밍 (MJPEG)
# ============================================================
# [2026-02-21 수정] 기존 YOLO만 → YOLO + BoT-SORT 트래킹 + 듀얼 카메라 전환
def generate_tracking_stream():
    """YOLO + BoT-SORT 트래킹 결과를 MJPEG 스트림으로 생성하는 제너레이터."""
    while True:
        try:
            # ── 현재 카메라 뷰에 따라 프레임 선택 ──
            with frame_lock:
                f_front = frame_front.copy() if frame_front is not None else None
                f_top = frame_top.copy() if frame_top is not None else None

            cam = current_camera                                    # 현재 카메라 모드

            if cam == "front":
                frame = f_front if f_front is not None else make_placeholder("FRONT: No signal")
            elif cam == "top":
                frame = f_top if f_top is not None else make_placeholder("TOP: No signal")
            else:
                # DUAL 모드: FRONT에 트래킹, TOP은 원본 나란히
                frame = f_front if f_front is not None else make_placeholder("FRONT: No signal")

            # ── YOLO + 트래킹 실행 (락으로 동기화) ──
            with track_lock:
                t0 = time.time()
                results = yolo_model.track(
                    frame, persist=True, conf=CONFIDENCE,
                    imgsz=YOLO_IMGSZ, tracker=current_tracker_yaml,
                    verbose=False
                )
                track_ms = (time.time() - t0) * 1000
                active = draw_tracking_results(frame, results)

            # ── DUAL 모드: TOP 카메라를 오른쪽에 합치기 ──
            if cam == "dual":
                top_frame = f_top if f_top is not None else make_placeholder("TOP: No signal")
                cv2.putText(top_frame, "TOP CAM (raw)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                frame = np.hstack([frame, top_frame])               # 1280x480

            # ── HUD 오버레이 ──
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

            tracker_display = "BoT-SORT" if current_tracker_name == "botsort" else "ByteTrack"
            hud = (f"{tracker_display} | {cam.upper()} | "
                   f"Track:{track_ms:.0f}ms | Obj:{active} | IDs:{len(all_ids)}")
            cv2.putText(frame, hud, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # ── JPEG 인코딩 + MJPEG yield ──
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        except Exception as e:
            print(f"⚠️ 스트림 오류: {e}")
            time.sleep(0.1)                                         # 폭주 방지

        time.sleep(0.03)                                            # ~30fps 제한


# ============================================================
# 5. Flask 라우트
# ============================================================

@app.route('/')
def index():
    """메인 페이지 — 마법사 UI HTML을 렌더링한다."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """YOLO + BoT-SORT 실시간 MJPEG 스트림."""
    return Response(
        generate_tracking_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# [2026-02-21 추가] 카메라 전환 API
@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    """카메라 뷰를 전환한다 (front → top → dual → front).

    POST /switch_camera
    POST /switch_camera  {"camera": "front"}  (직접 지정도 가능)
    """
    global current_camera, trails, all_ids

    data = request.get_json(silent=True) or {}

    if "camera" in data:
        # 직접 지정
        current_camera = data["camera"]
    else:
        # 순환 전환
        cycle = {"front": "top", "top": "dual", "dual": "front"}
        current_camera = cycle.get(current_camera, "front")

    # 카메라 전환 시 트래커 상태 리셋 (다른 영상이므로)
    with track_lock:
        trails.clear()
        all_ids.clear()
        yolo_model.predictor = None                                 # 트래커 내부 상태 리셋

    print(f"📷 카메라 전환: {current_camera.upper()}")
    return jsonify({"camera": current_camera})


# [2026-02-21 추가] 트래커 전환 API
@app.route('/switch_tracker', methods=['POST'])
def switch_tracker():
    """트래커를 전환한다 (BoT-SORT ↔ ByteTrack).

    POST /switch_tracker
    POST /switch_tracker  {"tracker": "bytetrack"}  (직접 지정도 가능)
    """
    global current_tracker_name, current_tracker_yaml, trails, all_ids

    data = request.get_json(silent=True) or {}

    if "tracker" in data:
        current_tracker_name = data["tracker"]
    else:
        # 토글
        if current_tracker_name == "botsort":
            current_tracker_name = "bytetrack"
        else:
            current_tracker_name = "botsort"

    current_tracker_yaml = BOTSORT_YAML if current_tracker_name == "botsort" else BYTETRACK_YAML

    # ★ 트래커 전환 시 3가지 상태 초기화 필수! (이슈 14)
    with track_lock:
        trails.clear()                                              # 1. 궤적 초기화
        all_ids.clear()                                             # 2. ID 목록 초기화
        yolo_model.predictor = None                                 # 3. 내부 상태 리셋 (핵심!)

    tracker_display = "BoT-SORT" if current_tracker_name == "botsort" else "ByteTrack"
    print(f"🔄 트래커 전환: {tracker_display}")
    return jsonify({"tracker": current_tracker_name, "display": tracker_display})


@app.route('/snapshot')
def snapshot():
    """현재 활성 카메라의 스냅샷을 base64 JPEG로 반환한다."""
    with frame_lock:
        if current_camera == "top":
            frame = frame_top.copy() if frame_top is not None else None
        else:
            frame = frame_front.copy() if frame_front is not None else None

    if frame is None:
        return jsonify({"error": "카메라 프레임 없음"}), 503

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

    return jsonify({
        "image": b64,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "camera": current_camera,
        "timestamp": time.time()
    })


@app.route('/detect', methods=['POST'])
def detect():
    """Grounding DINO 탐지 API — 텍스트 프롬프트로 객체를 탐지한다.

    Request JSON:
        {"text": "red bottle.", "box_threshold": 0.35, "text_threshold": 0.25}

    Response JSON:
        {"detections": [...], "inference_ms": 1234, "action": "...", "image": "base64..."}
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "텍스트 프롬프트 필요"}), 400

    text_prompt = data['text'].strip().lower()
    box_thresh = data.get('box_threshold', GDINO_BOX_THRESHOLD)
    text_thresh = data.get('text_threshold', GDINO_TEXT_THRESHOLD)

    # 마침표 자동 추가
    if text_prompt and not text_prompt.endswith('.'):
        text_prompt += '.'

    # [2026-02-21 수정] 현재 활성 카메라에서 프레임 가져오기
    with frame_lock:
        if current_camera == "top":
            frame = frame_top.copy() if frame_top is not None else None
        else:
            frame = frame_front.copy() if frame_front is not None else None

    if frame is None:
        return jsonify({"error": "카메라 프레임 없음. Pi에서 send_camera 실행 확인"}), 503

    # Grounding DINO 추론 (동시 요청 방지)
    with gdino_lock:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        inputs = gdino_processor(
            images=image_pil, text=text_prompt, return_tensors="pt"
        ).to("cpu")

        t0 = time.time()
        with torch.no_grad():
            outputs = gdino_model(**inputs)
        inference_ms = (time.time() - t0) * 1000

        # ⚠️ transformers 4.48+: box_threshold → threshold (이름 변경됨)
        results = gdino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[image_pil.size[::-1]]
        )

    result = results[0]
    boxes = result["boxes"]
    labels = result["labels"]
    scores = result["scores"]

    # [2026-06-20 추가] SAM3 버튼 클릭 시 재사용할 수 있도록 프레임+박스 저장
    global last_gdino_frame, last_gdino_boxes, last_gdino_prompt
    with last_gdino_lock:
        last_gdino_frame = frame.copy()                         # 탐지한 원본 프레임
        last_gdino_boxes = [box.tolist() for box in boxes]      # 박스 리스트
        last_gdino_prompt = text_prompt                         # 사용한 프롬프트

    annotated = frame.copy()
    detection_list = []

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = map(int, box.tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 보라색 박스로 GDINO 결과 표시
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (237, 58, 124), 3)
        cv2.putText(annotated, f"{label} {score:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (237, 58, 124), 2)

        detection_list.append({
            "label": label,
            "score": round(float(score), 4),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center": {"x": cx, "y": cy},
            "size": {"w": x2 - x1, "h": y2 - y1}
        })

    cv2.putText(annotated, f"GDINO: {inference_ms:.0f}ms | \"{text_prompt}\"",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (237, 58, 124), 2)

    _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    result_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

    action = compute_action(boxes, frame.shape[1])

    print(f"🗣️ GDINO: \"{text_prompt}\" → {len(detection_list)}개 탐지, "
          f"{inference_ms:.0f}ms, 행동={action}")

    return jsonify({
        "text_prompt": text_prompt,
        "detections": detection_list,
        "count": len(detection_list),
        "inference_ms": round(inference_ms, 1),
        "action": action,
        "image": result_b64,
        "camera": current_camera,
        "frame_size": {"w": frame.shape[1], "h": frame.shape[0]},
        "has_boxes": len(detection_list) > 0,          # [2026-06-20 추가] SAM3 버튼 활성화 여부
        "sam3_enabled": SAM3_ENABLED                   # [2026-06-20 추가] SAM3 기능 켜짐 여부
    })


@app.route('/segment_last', methods=['POST'])
def segment_last():
    """[2026-06-20 추가] 마지막 GDINO 탐지 결과를 4090 SAM3로 보내 정밀 마스크를 받는다.

    UI의 "SAM3 정밀분할" 버튼이 호출. /detect를 먼저 실행해 박스가 있어야 동작한다.

    Response JSON:
        {"box_image": base64, "mask_image": base64, "objects": [...],
         "precision_pct": N, "background_pct": N, "round_trip_ms": ms}
    """
    if not SAM3_ENABLED:
        return jsonify({"error": "SAM3 기능이 비활성화되어 있습니다 (SAM3_ENABLED=False)"}), 400

    # 마지막 GDINO 결과 가져오기
    with last_gdino_lock:
        frame = last_gdino_frame.copy() if last_gdino_frame is not None else None
        boxes = list(last_gdino_boxes)
        prompt = last_gdino_prompt

    if frame is None or len(boxes) == 0:
        return jsonify({"error": "먼저 텍스트로 물체를 탐지하세요 (박스 없음)"}), 400

    # 4090 SAM3 서버 호출
    t0 = time.time()
    sam3_resp = request_sam3_masks(frame, boxes, prompt)       # 프레임+박스+프롬프트 전송
    round_trip_ms = (time.time() - t0) * 1000

    # 실패 시 — 구체적 원인을 화면에 그대로 전달 (진단 쉽게)
    if sam3_resp is None or "_error" in sam3_resp:
        detail = sam3_resp.get("_error", "알 수 없는 오류") if sam3_resp else "응답 없음"
        return jsonify({"error": f"SAM3 실패 — {detail}"}), 503

    print(f"🎯 SAM3: \"{prompt}\" {len(boxes)}박스 → "
          f"{sam3_resp.get('count', 0)}마스크, 정확도 {sam3_resp.get('precision_pct', 0)}%, "
          f"왕복 {round_trip_ms:.0f}ms")

    return jsonify({
        "prompt": prompt,
        "box_image": sam3_resp.get("box_image"),               # 왼쪽: 박스 비교 이미지
        "mask_image": sam3_resp.get("mask_image"),             # 오른쪽: 마스크 비교 이미지
        "objects": sam3_resp.get("objects", []),               # 물체별 면적/정확도
        "count": sam3_resp.get("count", 0),
        "total_box_area": sam3_resp.get("total_box_area", 0),
        "total_mask_area": sam3_resp.get("total_mask_area", 0),
        "precision_pct": sam3_resp.get("precision_pct", 0),    # 전체 정확도
        "background_pct": sam3_resp.get("background_pct", 0),  # 제거된 배경
        "sam3_ms": sam3_resp.get("inference_ms", 0),           # 4090 순수 추론 시간
        "round_trip_ms": round(round_trip_ms, 1)               # NUC↔4090 왕복
    })


@app.route('/status')
def status():
    """서버 상태 확인 API."""
    with frame_lock:
        has_front = frame_front is not None
        has_top = frame_top is not None

    tracker_display = "BoT-SORT" if current_tracker_name == "botsort" else "ByteTrack"

    # [2026-06-20 추가] 4090 SAM3 서버 연결 확인 (1초 timeout 헬스체크)
    sam3_online = False
    if SAM3_ENABLED:
        try:
            r = requests.get(f"{SAM3_SERVER_URL}/ping", timeout=1.0)
            sam3_online = (r.status_code == 200)
        except requests.exceptions.RequestException:
            sam3_online = False

    return jsonify({
        "server": "running",
        "camera_front": has_front,
        "camera_top": has_top,
        "current_camera": current_camera,
        "current_tracker": tracker_display,
        "yolo_loaded": yolo_model is not None,
        "gdino_loaded": gdino_model is not None,
        "sam3_enabled": SAM3_ENABLED,                  # [2026-06-20 추가]
        "sam3_online": sam3_online,                    # [2026-06-20 추가] 4090 연결 여부
        "active_ids": len(all_ids),
        "pi_ip": PI_IP
    })


# ============================================================
# 6. 행동 결정 유틸리티
# ============================================================
def compute_action(boxes: torch.Tensor, frame_width: int = 640) -> str:
    """탐지된 박스 중 가장 큰 것의 중심 좌표로 행동을 결정한다."""
    if len(boxes) == 0:
        return "not_found"

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = areas.argmax()
    best_box = boxes[best_idx]
    cx = float((best_box[0] + best_box[2]) / 2)

    third = frame_width / 3
    if cx < third:
        return "move_left"
    elif cx > third * 2:
        return "move_right"
    else:
        return "move_forward"


# ============================================================
# 7. 서버 시작
# ============================================================
if __name__ == '__main__':
    print("=" * 65)
    print("🧙‍♂️ AI 마법사의 숨은물체 찾기 — 듀얼 카메라 + BoT-SORT")
    print("=" * 65)

    # BoT-SORT 설정 파일 생성
    if not os.path.exists(BOTSORT_YAML):
        create_botsort_yaml(BOTSORT_YAML)

    # 모델 로딩
    load_models()

    # [2026-02-21 수정] ZeroMQ 수신 스레드 2개 (FRONT + TOP)
    t_front = threading.Thread(
        target=zmq_receiver_thread,
        args=(FRONT_PORT, "front"),
        daemon=True
    )
    t_top = threading.Thread(
        target=zmq_receiver_thread,
        args=(TOP_PORT, "top"),
        daemon=True
    )
    t_front.start()
    t_top.start()

    print(f"\n🌐 웹 서버 시작: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"   브라우저에서 http://NUC_IP:{FLASK_PORT} 으로 접속하세요")
    print(f"\n📡 API 엔드포인트:")
    print(f"   GET  /video_feed       — YOLO + BoT-SORT MJPEG 스트림")
    print(f"   POST /switch_camera    — 카메라 전환 (FRONT/TOP/DUAL)")
    print(f"   POST /switch_tracker   — 트래커 전환 (BoT-SORT/ByteTrack)")
    print(f"   POST /detect           — Grounding DINO 언어 탐지")
    print(f"   POST /segment_last     — SAM3 정밀분할 (마지막 GDINO 박스 → 4090)")
    print(f"   GET  /snapshot         — 현재 프레임 스냅샷")
    print(f"   GET  /status           — 서버 상태")
    print(f"\n🎯 SAM3 연동: {SAM3_SERVER_URL} (UI에서 'SAM3 정밀분할' 버튼)")
    print("=" * 65)

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
