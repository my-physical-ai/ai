# NUC 오케스트레이터 — 시나리오 3: AI 물류 품질검사관 (v2 재설계)
# [2026-06-21 재설계] 파이프라인 순서 변경: YOLO→SAM3→VLM
#
# ★ 올바른 역할 분담:
#   YOLO (NUC CPU)  : "뭐가 어디에 있는지" 찾기 (33ms, 실시간)
#   SAM3 (4090 GPU) : YOLO 박스 → 정밀 픽셀 마스크 (윤곽선)
#   VLM  (4070 GPU) : 크롭된 이미지 → "손상 있나? 스크래치 있나?" 판단
#
# ★ 핵심 변경점 (v1 대비):
#   v1: YOLO → VLM(좌표 출력 시도) → SAM3 → 실패!
#   v2: YOLO(박스) → SAM3(마스크) → VLM(손상 판단) → 성공!
#   VLM에게 좌표를 요구하지 않음. VLM은 "판단"만 함.

import base64
import time
import threading

import cv2
import zmq
import numpy as np
import requests
from flask import Flask, render_template, Response, jsonify, request as flask_request
from ultralytics import YOLO

# ============================================================
# ★ 사용자 환경에 맞게 수정할 설정값들
# ============================================================
PI_IP = "192.168.50.111"                               # ← Pi5 실제 IP
PI_PORT = 5556                                          # ← Pi5 ZeroMQ 포트
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000                                       # NUC 웹서버 포트

# YOLOE-26 OpenVINO 모델 (NUC CPU 로컬)
# [2026-06-21 변경] YOLO26(80종 고정) → YOLOE-26(Open Vocabulary)
# setup_yoloe.py를 먼저 실행하여 모델 변환 필요!
YOLO_MODEL_PATH = "/home/zeta/vlm/yoloe-26s-seg_openvino_model/"   # ← setup_yoloe.py 출력 경로
YOLO_IMGSZ = 640

# 원격 AI 서버 주소
VLM_SERVER_URL = "http://192.168.0.36:5002"            # ← 4070 VLM 서버 IP:포트
SAM3_SERVER_URL = "http://192.168.0.75:5001"           # ← 4090 SAM3 서버 IP:포트

# HTTP 요청 타임아웃 (초)
VLM_TIMEOUT = 30
SAM3_TIMEOUT = 15

# [2026-06-21 추가] SAM3 라벨 매핑 테이블
# YOLOE가 탐지한 라벨 → SAM3가 알아듣는 일반 영어 단어로 변환
# 공장에서는 이 테이블을 "우리 제품 → SAM3 인식 단어"로 맞추면 됨
SAM3_LABEL_MAP = {
    # 음료 용기류 → "bottle" (SAM3가 잘 인식하는 단어)
    "tumbler": "bottle",
    "water bottle": "bottle",
    "coffee cup": "cup",
    "mug": "cup",
    "glass": "cup",
    "jar": "bottle",
    "can": "bottle",

    # 포장/상자류 → "box"
    "package": "box",
    "parcel": "box",
    "carton": "box",
    "crate": "box",
    "envelope": "paper",
    "container": "box",

    # 전자/산업류
    "circuit board": "board",
    "keyboard": "keyboard",
    "monitor": "screen",
    "laptop": "laptop",
    "phone": "phone",

    # 공장 부품류 → "object" (범용)
    "connector": "object",
    "bearing": "object",
    "gear": "object",
    "spring": "object",
    "gasket": "object",
    "tool": "tool",
    "wire": "wire",
    "pipe": "pipe",
    "helmet": "helmet",
}


def get_sam3_prompt(yolo_label: str) -> str:
    """YOLOE 라벨을 SAM3가 인식하는 영어 단어로 변환한다.

    [2026-06-21 추가] 공장 배포 시 이 매핑 테이블만 수정하면 됨.

    Args:
        yolo_label: YOLOE가 탐지한 영어 라벨 ("tumbler", "package")

    Returns:
        SAM3용 영어 프롬프트 ("bottle", "box")
    """
    label_lower = yolo_label.lower().strip()

    # 매핑 테이블에 있으면 변환
    if label_lower in SAM3_LABEL_MAP:
        mapped = SAM3_LABEL_MAP[label_lower]
        print(f"   🔄 SAM3 라벨 매핑: \"{label_lower}\" → \"{mapped}\"")
        return mapped

    # 없으면 원래 라벨 그대로 사용
    return label_lower

# ============================================================
# 전역 상태
# ============================================================
app = Flask(__name__)
latest_frame = None
latest_frame_lock = threading.Lock()
yolo_model = None


# ============================================================
# 1. ZeroMQ 프레임 수신 스레드
# ============================================================
def zmq_receiver_thread():
    """Pi 카메라 프레임을 백그라운드에서 계속 수신한다."""
    global latest_frame
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.connect(f"tcp://{PI_IP}:{PI_PORT}")
    print(f"📡 ZeroMQ 수신 시작: tcp://{PI_IP}:{PI_PORT}")

    while True:
        try:
            buf = sock.recv()
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), 1)
            if frame is not None:
                with latest_frame_lock:
                    latest_frame = frame
        except zmq.Again:
            pass
        except Exception as e:
            print(f"❌ ZeroMQ 오류: {e}")
            time.sleep(1)


# ============================================================
# 2. YOLO MJPEG 스트리밍
# ============================================================
def generate_yolo_stream():
    """YOLO 탐지 결과를 MJPEG 스트림으로 생성한다."""
    while True:
        with latest_frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera...",
                        (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            frame = placeholder

        results = yolo_model(frame, verbose=False, imgsz=YOLO_IMGSZ)
        annotated = results[0].plot()
        cv2.putText(annotated, "YOLO26 | Live", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)


# ============================================================
# 3. 유틸리티 함수
# ============================================================
def frame_to_base64(frame_bgr: np.ndarray) -> str:
    """BGR 프레임을 base64 JPEG 문자열로 변환한다."""
    _, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')


def run_yolo_detection(frame: np.ndarray) -> list:
    """YOLO로 물체를 탐지하고 결과 리스트를 반환한다.

    [2026-06-21 추가] VLM 대신 YOLO가 물체 위치를 찾는다.

    Returns:
        [{"label": "cup", "confidence": 0.87, "box": [x1,y1,x2,y2]}, ...]
    """
    results = yolo_model(frame, verbose=False, imgsz=YOLO_IMGSZ)
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "box": [x1, y1, x2, y2]
            })

    # 신뢰도 높은 순으로 정렬
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def crop_region(frame: np.ndarray, box: list, padding: int = 20) -> np.ndarray:
    """바운딩 박스 영역을 패딩 포함해서 크롭한다.

    Args:
        frame: 원본 프레임 (BGR)
        box: [x1, y1, x2, y2]
        padding: 박스 주변 여백 (픽셀)

    Returns:
        크롭된 이미지 (BGR)
    """
    h, w = frame.shape[:2]
    x1 = max(0, box[0] - padding)
    y1 = max(0, box[1] - padding)
    x2 = min(w, box[2] + padding)
    y2 = min(h, box[3] + padding)
    return frame[y1:y2, x1:x2].copy()


def call_sam3(image_b64: str, boxes: list, prompt: str = "object") -> dict:
    """4090 SAM3 서버에 정밀 세그멘테이션 요청을 보낸다.

    [2026-06-21 재설계] YOLO 박스 + YOLO 라벨을 SAM3에 전달.
    SAM3 서버가 set_text_prompt 방식이므로 YOLO 영어 라벨을 prompt로 사용.

    Args:
        image_b64: 원본 이미지 base64
        boxes: YOLO 바운딩 박스 [[x1,y1,x2,y2], ...]
        prompt: YOLO가 탐지한 영어 라벨 (예: "cup", "bottle")
    """
    print(f"   📡 SAM3 요청: boxes={len(boxes)}, prompt=\"{prompt}\"")

    try:
        resp = requests.post(
            f"{SAM3_SERVER_URL}/segment",
            json={
                "image": image_b64,
                "boxes": boxes,
                "prompt": prompt,
            },
            timeout=SAM3_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "SAM3 서버 연결 실패 — 4090 서버 확인 필요"}
    except requests.exceptions.Timeout:
        return {"error": f"SAM3 서버 타임아웃 ({SAM3_TIMEOUT}초)"}
    except Exception as e:
        return {"error": f"SAM3 호출 실패: {str(e)}"}


def call_vlm_judge(image_b64: str, label: str, user_prompt: str = "") -> dict:
    """4070 VLM 서버에 품질 판단 요청을 보낸다.

    [2026-06-21 재설계] VLM은 좌표를 찾지 않는다.
    크롭된 이미지를 받아 "손상 여부"만 판단한다.

    Args:
        image_b64: 크롭된 물체 이미지 base64
        label: YOLO가 탐지한 영어 라벨 ("cup", "bottle")
        user_prompt: 사용자 추가 질문 (선택)
    """
    # [2026-06-21 추가] VLM 전송 전 이미지 최소 크기 보장
    # Qwen2.5-VL은 이미지를 패치로 나눠 분석 → 작은 이미지는 디테일 손실
    # 최소 672x672로 확대하여 충분한 패치 수 확보
    import cv2 as cv2_local
    img_bytes = base64.b64decode(image_b64)
    img_arr = cv2_local.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
    h, w = img_arr.shape[:2]
    MIN_SIZE = 672
    if max(h, w) < MIN_SIZE:
        scale = MIN_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_arr = cv2_local.resize(img_arr, (new_w, new_h), interpolation=cv2_local.INTER_LANCZOS4)
        _, jpeg = cv2_local.imencode('.jpg', img_arr, [cv2_local.IMWRITE_JPEG_QUALITY, 95])
        image_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        print(f"      📐 이미지 확대: {w}x{h} → {new_w}x{new_h} (VLM 패치 확보)")

    # [2026-06-21 수정] VLM에게 "정상 상태 기준"을 알려줘야 찌그러짐을 감지함
    # VLM은 "찌그러진 병"을 봐도 "이게 원래 이런 건가?" 판단 불가 → 기준이 필요!
    prompt = (
        f"당신은 엄격한 품질검사관입니다.\n\n"
        f"[정상 기준] 정상적인 '{label}'은:\n"
        f"- 표면이 매끄럽고 주름이나 울퉁불퉁함이 없습니다\n"
        f"- 좌우 대칭이고 원래 형태를 유지합니다\n"
        f"- 찌그러짐, 눌림, 구겨짐이 없습니다\n"
        f"- 스크래치나 긁힌 자국이 없습니다\n\n"
        f"[무시할 것] 아래는 정상적인 제조 특성이므로 결함이 아닙니다:\n"
        f"- 금형 자국, 접합선 (mold line)\n"
        f"- 엠보싱, 로고, 눈금 표시\n"
        f"- 라벨 질감, 인쇄 패턴\n"
        f"- 병 바닥이나 뚜껑의 제조 흔적\n\n"
        f"[검사 대상] 이 사진에서 아래 '실제 결함'만 찾으세요:\n"
        f"- 찌그러짐, 눌림, 구겨짐 (형태 변형)\n"
        f"- 스크래치, 긁힘 (표면 손상)\n"
        f"- 깨짐, 크랙 (파손)\n"
        f"- 이물질, 오염\n\n"
        f"제조 특성과 실제 결함을 구분하세요.\n"
        f"최종 판정: 양품 / 주의 / 불량\n"
    )
    if user_prompt:
        prompt += f"\n⚠️ 특별 지시: {user_prompt}\n이 부분을 특히 주의해서 확인하세요."

    try:
        resp = requests.post(
            f"{VLM_SERVER_URL}/chat",
            json={"image": image_b64, "prompt": prompt},
            timeout=VLM_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "VLM 서버 연결 실패 — 4070 서버 확인 필요"}
    except requests.exceptions.Timeout:
        return {"error": f"VLM 서버 타임아웃 ({VLM_TIMEOUT}초)"}
    except Exception as e:
        return {"error": f"VLM 호출 실패: {str(e)}"}


def call_vlm_chat(image_b64: str, prompt: str) -> dict:
    """4070 VLM 서버에 자유 대화 요청을 보낸다."""
    try:
        resp = requests.post(
            f"{VLM_SERVER_URL}/chat",
            json={"image": image_b64, "prompt": prompt},
            timeout=VLM_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"VLM 대화 실패: {str(e)}"}


# ============================================================
# 4. Flask 라우트
# ============================================================
@app.route('/')
def index():
    return render_template('scenario3.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_yolo_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    with latest_frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "카메라 프레임 없음"}), 503
    b64 = frame_to_base64(frame)
    return jsonify({"image": b64, "w": frame.shape[1], "h": frame.shape[0]})


@app.route('/inspect', methods=['POST'])
def inspect():
    """품질 검사 실행 — 재설계된 3단계 파이프라인.

    [2026-06-21 재설계] YOLO → SAM3 → VLM 순서:
      Step 1: YOLO가 물체를 찾는다 (NUC 로컬, 33ms)
      Step 2: SAM3가 YOLO 박스로 정밀 마스크를 만든다 (4090)
      Step 3: VLM이 크롭된 이미지로 손상 여부를 판단한다 (4070)

    Request JSON:
        {"prompt": "스크래치 확인해주세요", "target": "cup"} (선택)
        target: 특정 물체만 검사 (없으면 가장 큰/신뢰도 높은 물체)
    """
    data = flask_request.get_json() or {}
    user_prompt = data.get('prompt', '')
    target_label = data.get('target', '')

    t_total = time.time()

    # ── Step 1: 프레임 캡처 + YOLO 탐지 (NUC 로컬) ──
    with latest_frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "카메라 프레임 없음"}), 503

    image_b64 = frame_to_base64(frame)

    print(f"\n{'='*50}")
    print(f"🔍 품질 검사 시작 (v2: YOLO→SAM3→VLM)")

    t0 = time.time()
    detections = run_yolo_detection(frame)
    yolo_ms = (time.time() - t0) * 1000

    print(f"   Step 1 YOLO: {len(detections)}개 탐지, {yolo_ms:.0f}ms")
    for d in detections:
        print(f"      - {d['label']} ({d['confidence']:.2f}) box={d['box']}")

    if len(detections) == 0:
        return jsonify({
            "yolo": {"detections": [], "count": 0, "ms": round(yolo_ms, 1)},
            "sam3": {"message": "YOLO 탐지 없음"},
            "vlm": {"message": "검사할 물체 없음"},
            "snapshot": image_b64,
            "total_ms": round((time.time() - t_total) * 1000, 1)
        })

    # 검사 대상 선택: target 지정 시 해당 라벨, 아니면 가장 신뢰도 높은 것
    if target_label:
        matched = [d for d in detections if d['label'].lower() == target_label.lower()]
        target = matched[0] if matched else detections[0]
    else:
        target = detections[0]

    print(f"   → 검사 대상: {target['label']} ({target['confidence']:.2f})")

    # ── Step 2: SAM3 정밀 마스크 (4090) ──
    # [2026-06-21 수정] YOLOE 라벨을 SAM3가 인식하는 단어로 변환!
    sam3_label = get_sam3_prompt(target['label'])
    print(f"   Step 2 SAM3: \"{target['label']}\" → \"{sam3_label}\" (매핑)")
    sam3_result = call_sam3(image_b64, [target['box']], prompt=sam3_label)
    sam3_ms = sam3_result.get('inference_ms', 0)

    if "error" in sam3_result:
        print(f"   ⚠️ SAM3 에러: {sam3_result['error']}")
    else:
        print(f"   → SAM3: 정확도 {sam3_result.get('precision_pct', 0):.0f}%, {sam3_ms:.0f}ms")

    # ── Step 3: VLM 품질 판단 (4070) ──
    # [2026-06-21 수정] SAM3 cutout에서 물체만 타이트하게 크롭 → 확대 → VLM
    # 배경(흰색) 제거 후 물체 영역만 꽉 차게 확대하면 표면 디테일이 극대화됨
    cutout_b64 = sam3_result.get("cutout_image", "")
    vlm_image = ""

    if cutout_b64:
        # cutout(흰배경+물체)에서 물체 영역만 타이트하게 잘라서 확대
        import cv2 as cv2_local
        cut_bytes = base64.b64decode(cutout_b64)
        cut_img = cv2_local.imdecode(np.frombuffer(cut_bytes, np.uint8), 1)

        # 흰색이 아닌 영역(물체) 찾기
        gray = cv2_local.cvtColor(cut_img, cv2_local.COLOR_BGR2GRAY)
        mask = gray < 250                                   # 흰색(255)이 아닌 픽셀
        coords = np.argwhere(mask)                           # 물체 픽셀 좌표

        if len(coords) > 100:                                # 물체가 충분히 있으면
            y1, x1 = coords.min(axis=0)                      # 타이트한 바운딩 박스
            y2, x2 = coords.max(axis=0)

            # 약간의 여백 추가
            pad = 10
            y1 = max(0, y1 - pad)
            x1 = max(0, x1 - pad)
            y2 = min(cut_img.shape[0], y2 + pad)
            x2 = min(cut_img.shape[1], x2 + pad)

            # 타이트하게 크롭
            tight_crop = cut_img[y1:y2, x1:x2]

            # 2048px로 확대 (Lanczos — 최대 디테일)
            h_c, w_c = tight_crop.shape[:2]
            TARGET = 1024
            scale = TARGET / max(h_c, w_c)
            new_w = int(w_c * scale)
            new_h = int(h_c * scale)
            zoomed = cv2_local.resize(tight_crop, (new_w, new_h),
                                       interpolation=cv2_local.INTER_LANCZOS4)

            _, jpeg = cv2_local.imencode('.jpg', zoomed, [cv2_local.IMWRITE_JPEG_QUALITY, 95])
            vlm_image = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            print(f"   Step 3 VLM: SAM3 cutout 확대! {w_c}x{h_c} → {new_w}x{new_h}px")
        else:
            print(f"   Step 3 VLM: cutout에서 물체 영역 부족 → 직사각형 크롭 사용")

    if not vlm_image:
        crop = crop_region(frame, target['box'])
        vlm_image = frame_to_base64(crop)
        print(f"   Step 3 VLM: 직사각형 크롭 사용 (SAM3 cutout 없음)")

    print(f"   Step 3 VLM: 품질 판단 요청")
    vlm_result = call_vlm_judge(vlm_image, target['label'], user_prompt)
    vlm_ms = vlm_result.get('inference_ms', 0)

    if "error" in vlm_result:
        print(f"   ⚠️ VLM 에러: {vlm_result['error']}")
    else:
        print(f"   → VLM: {vlm_ms:.0f}ms")
        answer_preview = vlm_result.get('answer', '')[:60]
        print(f"   → 판단: {answer_preview}...")

    total_ms = (time.time() - t_total) * 1000
    print(f"📊 전체 파이프라인: {total_ms:.0f}ms (YOLO {yolo_ms:.0f} + SAM3 {sam3_ms:.0f} + VLM {vlm_ms:.0f})")
    print(f"{'='*50}")

    return jsonify({
        "yolo": {
            "detections": detections,
            "target": target,
            "count": len(detections),
            "ms": round(yolo_ms, 1)
        },
        "sam3": sam3_result,
        "vlm": vlm_result,
        "snapshot": image_b64,
        "crop": cutout_b64 if cutout_b64 else frame_to_base64(crop_region(frame, target['box'])),
        "total_ms": round(total_ms, 1)
    })


@app.route('/inspect_precision', methods=['POST'])
def inspect_precision():
    """정밀 검사 v3 — 물체를 상/중/하 3등분하여 각각 VLM 검사.

    [2026-06-21 추가] VLM이 전체 이미지를 한 번에 보면 미세 결함을 놓침.
    3등분하여 각 영역을 개별 검사하면 한쪽만 찌그러진 것도 감지 가능.

    소요 시간: VLM x3회 호출 (약 45~60초)
    """
    data = flask_request.get_json() or {}
    user_prompt = data.get('prompt', '')

    t_total = time.time()

    with latest_frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "카메라 프레임 없음"}), 503

    image_b64 = frame_to_base64(frame)

    print(f"\n{'='*50}")
    print(f"🔬 정밀 검사 v3 (3등분 모드: YOLO→SAM3→VLM×3)")

    # ── Step 1: YOLOE 탐지 ──
    t0 = time.time()
    detections = run_yolo_detection(frame)
    yolo_ms = (time.time() - t0) * 1000

    if not detections:
        return jsonify({"error": "YOLO 탐지 없음", "yolo": {"count": 0}})

    target = detections[0]
    print(f"   Step 1 YOLOE: {target['label']} ({target['confidence']:.2f})")

    # ── Step 2: SAM3 마스크 ──
    sam3_label = get_sam3_prompt(target['label'])
    sam3_result = call_sam3(image_b64, [target['box']], prompt=sam3_label)

    # ── Step 3: SAM3 cutout → 3등분 → VLM 각각 검사 ──
    cutout_b64 = sam3_result.get("cutout_image", "")
    zone_results = []
    zone_images = []
    zone_names = ["상단 (위쪽)", "중앙 (가운데)", "하단 (아래쪽)"]

    if cutout_b64:
        import cv2 as cv2_local

        # cutout에서 물체 영역 추출
        cut_bytes = base64.b64decode(cutout_b64)
        cut_img = cv2_local.imdecode(np.frombuffer(cut_bytes, np.uint8), 1)
        gray = cv2_local.cvtColor(cut_img, cv2_local.COLOR_BGR2GRAY)
        mask = gray < 250
        coords = np.argwhere(mask)

        if len(coords) > 100:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            pad = 5
            y1 = max(0, y1 - pad)
            x1 = max(0, x1 - pad)
            y2 = min(cut_img.shape[0], y2 + pad)
            x2 = min(cut_img.shape[1], x2 + pad)

            tight = cut_img[y1:y2, x1:x2]
            h_t = tight.shape[0]

            # 3등분 (상/중/하)
            zones = [
                tight[0:h_t//3, :],                          # 상단
                tight[h_t//3:2*h_t//3, :],                   # 중앙
                tight[2*h_t//3:, :],                          # 하단
            ]

            for i, (zone_img, zone_name) in enumerate(zip(zones, zone_names)):
                # 각 영역을 1024px로 확대
                zh, zw = zone_img.shape[:2]
                scale = 1024 / max(zh, zw)
                z_resized = cv2_local.resize(zone_img,
                    (int(zw * scale), int(zh * scale)),
                    interpolation=cv2_local.INTER_LANCZOS4)

                _, z_jpeg = cv2_local.imencode('.jpg', z_resized,
                    [cv2_local.IMWRITE_JPEG_QUALITY, 95])
                z_b64 = base64.b64encode(z_jpeg.tobytes()).decode('utf-8')
                zone_images.append(z_b64)

                # 각 영역별 VLM 검사
                print(f"   Step 3-{i+1} VLM: {zone_name} 검사 중...")
                z_prompt = (
                    f"당신은 엄격한 품질검사관입니다.\n"
                    f"이것은 '{target['label']}'의 {zone_name} 부분을 확대한 사진입니다.\n\n"
                    f"[무시] 금형 자국, 엠보싱, 로고, 라벨 질감은 정상 제조 특성입니다.\n"
                    f"[검사] 아래 '실제 결함'만 찾으세요:\n"
                    f"- 찌그러짐, 눌림, 구겨짐 (형태 변형)\n"
                    f"- 스크래치, 긁힘 (표면 손상)\n"
                    f"- 깨짐, 크랙, 이물질\n\n"
                    f"제조 특성과 실제 결함을 구분하세요.\n"
                    f"판정: 양품 / 주의 / 불량\n"
                )
                if user_prompt:
                    z_prompt += f"\n⚠️ 특별 지시: {user_prompt}"

                try:
                    resp = requests.post(
                        f"{VLM_SERVER_URL}/chat",
                        json={"image": z_b64, "prompt": z_prompt},
                        timeout=VLM_TIMEOUT
                    )
                    resp.raise_for_status()
                    z_result = resp.json()
                except Exception as e:
                    z_result = {"error": str(e)}

                zone_results.append({
                    "zone": zone_name,
                    "answer": z_result.get("answer", z_result.get("error", "")),
                    "inference_ms": z_result.get("inference_ms", 0)
                })
                print(f"      → {zone_name}: {z_result.get('answer', '')[:40]}...")

    total_ms = (time.time() - t_total) * 1000
    print(f"📊 정밀 검사 완료: {total_ms:.0f}ms (VLM×3)")
    print(f"{'='*50}")

    return jsonify({
        "mode": "precision",
        "yolo": {"target": target, "count": len(detections), "ms": round(yolo_ms, 1)},
        "sam3": sam3_result,
        "vlm_zones": zone_results,
        "zone_images": zone_images,
        "snapshot": image_b64,
        "total_ms": round(total_ms, 1)
    })


@app.route('/chat', methods=['POST'])
def chat():
    """VLM 대화 — 검사 결과에 대해 질문한다."""
    data = flask_request.get_json() or {}
    prompt = data.get('prompt', '이 이미지를 설명해주세요.')

    with latest_frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "카메라 프레임 없음"}), 503

    image_b64 = frame_to_base64(frame)
    result = call_vlm_chat(image_b64, prompt)
    return jsonify(result)


@app.route('/status')
def status():
    """전체 시스템 상태 확인."""
    with latest_frame_lock:
        has_frame = latest_frame is not None

    vlm_ok, vlm_info = False, {}
    try:
        r = requests.get(f"{VLM_SERVER_URL}/ping", timeout=3)
        if r.status_code == 200:
            vlm_ok = True
            vlm_info = r.json()
    except Exception:
        pass

    sam3_ok, sam3_info = False, {}
    try:
        r = requests.get(f"{SAM3_SERVER_URL}/ping", timeout=3)
        if r.status_code == 200:
            sam3_ok = True
            sam3_info = r.json()
    except Exception:
        pass

    return jsonify({
        "camera": has_frame,
        "yolo": yolo_model is not None,
        "vlm": {"connected": vlm_ok, "url": VLM_SERVER_URL, **vlm_info},
        "sam3": {"connected": sam3_ok, "url": SAM3_SERVER_URL, **sam3_info}
    })


# ============================================================
# 5. 서버 시작
# ============================================================
if __name__ == '__main__':
    print("=" * 65)
    print("📦 AI 물류 품질검사관 v2 — YOLOE-26 → SAM3 → VLM")
    print("   YOLOE-26: Open Vocabulary (텍스트로 무엇이든 탐지)")
    print("=" * 65)

    print(f"⚡ YOLO 로딩: {YOLO_MODEL_PATH}")
    # [2026-06-21 변경] YOLOE-seg 모델은 탐지+세그멘테이션 모두 지원
    # task를 지정하지 않으면 모델에서 자동 감지
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO 준비 완료")

    receiver = threading.Thread(target=zmq_receiver_thread, daemon=True)
    receiver.start()

    print(f"\n🌐 웹 서버: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"   VLM 서버 (품질 판단): {VLM_SERVER_URL}")
    print(f"   SAM3 서버 (정밀 마스크): {SAM3_SERVER_URL}")
    print(f"\n📋 파이프라인: YOLO(NUC) → SAM3(4090) → VLM(4070)")
    print("=" * 65)

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
