# ============================================================
# 파일명: app.py
# 설명: 열화상 3등분 온도측정 AI — YOLO 실시간 탐지 + 상대지수 재정규화 + InternVL 판정
# 환경: NUC10i7FNH (192.168.0.65), conda lerobot, /dev/video3 (FLIR ONE Gen2 열화상)
# 변경 이력:
#   - [2026-06-23] inferno 역변환(thermal_level) + 박스 재정규화(zone_measure)
#                  + IRGPT 규칙기반 → InternVL 이미지 판정(vlm_judge) 전환
# ============================================================

import os                                            # 경로 처리
import time                                          # 시간 측정
import threading                                     # 멀티스레드 동기화

import cv2                                           # 영상 처리 (OpenCV)
import yaml                                          # 설정 파일 로드
import numpy as np                                  # 배열 연산
from flask import Flask, render_template, Response, jsonify, request  # 웹서버
from thermal_level import rgb_to_level_fast, rgb_to_level_grayscale  # 컬러/회색조 변환
from zone_measure import measure_zones_renorm, measure_zones_mask  # 박스/마스크 측정
from vlm_judge import judge_with_vlm, format_verdict  # InternVL 판정 (IRGPT 대체)
from sam3_client import (request_person_masks, check_sam3_online,  # 박스 분할
                         request_person_mask_text)  # 텍스트 분할 (사람 윤곽용)

# ============================================================
# 설정 로드 — app.py 위치 기준으로 config.yaml 자동 탐색
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py가 있는 폴더
_candidates = [                                       # config 후보 경로들
    os.path.join(BASE_DIR, "config", "config.yaml"),
    os.path.join(BASE_DIR, "..", "config", "config.yaml"),
]
CONFIG_PATH = next((p for p in _candidates if os.path.exists(p)), None)  # 존재 경로 선택
if CONFIG_PATH is None:                              # 못 찾으면 에러
    raise FileNotFoundError("config.yaml을 찾을 수 없습니다")
with open(CONFIG_PATH, encoding="utf-8") as f:       # 설정 파일 열기
    CFG = yaml.safe_load(f)                          # YAML 파싱
print(f"⚙️ 설정 로드: {CONFIG_PATH}")                # 로드 경로 출력 (디버깅용)

# 설정값 추출
DEVICE_INDEX = CFG["thermal"]["device_index"]        # 열화상 카메라 번호 (3)
DISPLAY_COLORMAP = CFG["thermal"].get("display_colormap", "inferno")  # 표시 색상
# flirone 팔레트 종류 (grayscale=회색조, color=컬러 inferno) — 카메라 운영 팔레트와 일치 필수
# ※ 화면이 컬러(inferno)이면 color, 회색조(Grayscale.raw)이면 grayscale
PALETTE_TYPE = CFG["thermal"].get("palette_type", "color")  # 기본 컬러 (현재 inferno 운영)
YOLO_MODEL_PATH = CFG["yolo"]["model_path"]          # YOLO 모델 경로
YOLO_IMGSZ = CFG["yolo"]["imgsz"]                    # 추론 입력 크기
YOLO_CONF = CFG["yolo"]["conf"]                      # 신뢰도 임계값
YOLO_IOU = CFG["yolo"].get("iou", 0.45)             # NMS 임계값
TARGET_CLASSES = set(CFG["yolo"]["target_classes"])  # 탐지 대상 클래스 (0=person)
GRAY_INPUT = CFG["yolo"].get("grayscale_input", True)  # YOLO 입력 흑백 변환 여부
MAX_PERSONS = CFG["yolo"].get("max_persons", 1)      # 최대 탐지 인원
ZONE_NAMES = CFG["zones"]["names"]                   # 구역 이름 [상,중,하]
ALERT_LEVEL = CFG["zones"].get("alert_level", 75)    # 고온 경고 기준 (상대지수)
# SAM3 윤곽 사용 여부 (config의 sam3.enabled, 4090 서버 필요)
SAM3_ENABLED = CFG.get("sam3", {}).get("enabled", True)  # 기본 사용 (4090 꺼지면 자동 폴백)
FLASK_HOST = CFG["flask"]["host"]                    # 서버 호스트
FLASK_PORT = CFG["flask"]["port"]                    # 서버 포트

# ============================================================
# Flask 앱 + 전역 상태
# ============================================================
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),  # HTML 폴더
            static_folder=os.path.join(BASE_DIR, "static"))       # 정적 파일 폴더

latest_vis = None            # 최신 의사색상 프레임 (BGR, 표시용)
latest_level = None          # 최신 상대온도 지수 배열 (0~100)
latest_boxes = []            # 최신 YOLO 박스 [[x1,y1,x2,y2,conf], ...]
frame_lock = threading.Lock()  # 프레임 접근 동기화
yolo_model = None            # YOLO 모델 (전역)
judge_result = {"zones": [], "analysis": "", "image": "", "ts": 0}  # 판별 결과 저장
# 단계별 캡처 결과 저장 (SAM3 → 세분화 → 판별 간 데이터 공유)
stage_state = {"vis": None, "level": None, "boxes": [],   # 캡처된 프레임/지수/박스
               "masks": None, "zones": None}              # SAM3 마스크/측정 결과
stage_lock = threading.Lock()  # 단계 상태 동기화
btn_lock = threading.Lock()  # 버튼 결과 동기화


# ============================================================
# YOLO 모델 로드 (시작 시 1회)
# ============================================================
def init_yolo():
    """열화상 YOLO26 OpenVINO 모델을 로드한다."""
    global yolo_model
    from ultralytics import YOLO                     # YOLO 라이브러리
    path = YOLO_MODEL_PATH                            # config의 모델 경로
    if not os.path.exists(path):                     # 경로 없으면 자동 탐색
        home = os.path.expanduser("~")               # 홈 디렉토리
        for c in [os.path.join(home, "thermal", "best_openvino_model"),
                  os.path.join(BASE_DIR, "..", "best_openvino_model")]:
            if os.path.exists(c):                    # 후보 경로 존재하면
                path = c                              # 그 경로 사용
                print(f"⚠️ config 경로 없음 → 자동 발견: {path}")
                break
    print(f"⚡ YOLO26 로딩: {path}")                  # 로딩 경로 출력
    yolo_model = YOLO(path, task="detect")           # 모델 로드 (탐지 모드)
    print("✅ YOLO 준비 완료")


# ============================================================
# 유틸 함수
# ============================================================
def yolo_input(vis):
    """YOLO 입력 영상을 흑백으로 변환한다 (학습 데이터가 흑백이라 색 맞춤)."""
    if not GRAY_INPUT:                               # 흑백 변환 안 하면
        return vis                                    # 원본 반환
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)     # 흑백 변환
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)    # 다시 3채널로 (모델 입력용)


def to_b64(img):
    """BGR 이미지를 base64 JPEG 문자열로 변환한다 (브라우저 표시용)."""
    import base64                                     # 이미지 인코딩
    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])  # JPEG 인코딩
    return base64.b64encode(jpeg.tobytes()).decode()  # base64 문자열 반환


def merge_boxes(boxes, iou_thr=0.15, contain_thr=0.5, max_persons=0):
    """겹치거나 포함되는 박스를 병합한다 (한 사람 다중박스 방지)."""
    if len(boxes) <= 1:                              # 1개 이하면 병합 불필요
        return boxes
    boxes = sorted(boxes, key=lambda b: -b[4])       # 신뢰도 높은 순 정렬
    keep = []                                         # 유지할 박스 목록
    for b in boxes:                                   # 각 박스 검사
        x1, y1, x2, y2 = b[:4]                        # 박스 좌표
        a1 = max(0, x2 - x1) * max(0, y2 - y1)        # 박스 면적
        dup = False                                   # 중복 여부
        for k in keep:                                # 이미 유지된 박스와 비교
            kx1, ky1, kx2, ky2 = k[:4]                # 비교 박스 좌표
            ix1, iy1 = max(x1, kx1), max(y1, ky1)     # 교집합 좌상단
            ix2, iy2 = min(x2, kx2), min(y2, ky2)     # 교집합 우하단
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)  # 교집합 면적
            a2 = max(0, kx2 - kx1) * max(0, ky2 - ky1)  # 비교 박스 면적
            union = a1 + a2 - inter                   # 합집합 면적
            iou = inter / union if union > 0 else 0   # IoU 계산
            contain = inter / min(a1, a2) if min(a1, a2) > 0 else 0  # 포함 비율
            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2   # 박스 중심
            cx2, cy2 = (kx1 + kx2) / 2, (ky1 + ky2) / 2  # 비교 박스 중심
            cdist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5  # 중심 거리
            center_close = cdist < max(x2 - x1, kx2 - kx1) * 0.5  # 중심 근접 여부
            if iou > iou_thr or contain > contain_thr or center_close:  # 중복 판정
                dup = True                            # 중복으로 표시
                break
        if not dup:                                   # 중복 아니면
            keep.append(b)                            # 유지 목록에 추가
    if max_persons > 0:                               # 최대 인원 제한 있으면
        keep = keep[:max_persons]                     # 상위 N명만
    return keep


# ============================================================
# 카메라 캡처 + YOLO 실시간 탐지 스레드
# ============================================================
def capture_thread():
    """열화상 카메라를 계속 읽고 YOLO 탐지를 실시간 수행한다."""
    global latest_vis, latest_level, latest_boxes
    cap = cv2.VideoCapture(DEVICE_INDEX)             # 카메라 열기
    if not cap.isOpened():                           # 못 열면 종료
        print(f"❌ /dev/video{DEVICE_INDEX} 열기 실패 — flirone 실행 확인")
        return
    print(f"📷 카메라 시작: /dev/video{DEVICE_INDEX} (상대온도 모드)")
    while True:                                       # 무한 루프
        ret, raw = cap.read()                        # 한 프레임 읽기
        if not ret or raw is None:                   # 실패하면
            time.sleep(0.05)                         # 잠깐 대기 후 재시도
            continue

        # ★팔레트 종류에 따라 온도지수 변환 (회색조=밝기 그대로, 컬러=inferno 역변환)
        if raw.ndim == 3:                            # RGB 입력이면
            if PALETTE_TYPE == "grayscale":          # 회색조 팔레트(Grayscale.raw)면
                level = rgb_to_level_grayscale(raw)  # 밝기 그대로 사용 (단순·정확)
            else:                                    # 컬러 팔레트(Iron/Inferno)면
                level = rgb_to_level_fast(raw)       # inferno 단조 역변환 (노랑>주황 역전 방지)
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)  # 표시용 흑백
        else:                                        # 이미 단일채널이면
            gray = raw                               # 그대로 사용
            level = gray.astype(np.float32) / 255.0 * 100.0  # 밝기=온도

        # 표시용 의사색상 (config에 따라 컬러 또는 흑백)
        norm8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 0~255 정규화
        if DISPLAY_COLORMAP == "gray":               # 흑백 표시 설정이면
            vis = cv2.cvtColor(norm8, cv2.COLOR_GRAY2BGR)  # 흑백 3채널
        else:                                         # 컬러 표시(기본)
            vis = cv2.applyColorMap(norm8, cv2.COLORMAP_INFERNO)  # inferno 컬러맵

        # 실시간 YOLO 탐지
        boxes = []                                    # 이번 프레임 박스 목록
        try:
            res = yolo_model(yolo_input(vis), verbose=False,  # 흑백 입력으로 추론
                             imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU)
            for b in res[0].boxes:                    # 탐지된 각 박스
                if TARGET_CLASSES and int(b.cls[0]) not in TARGET_CLASSES:  # 대상 아니면
                    continue                          # 건너뜀
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())  # 박스 좌표
                bw, bh = x2 - x1, y2 - y1             # 박스 크기
                if bw < 0.08 * 640 or bh < 0.12 * 480:  # 너무 작으면 노이즈
                    continue                          # 제외
                boxes.append([x1, y1, x2, y2, float(b.conf[0])])  # 박스 추가
            boxes = merge_boxes(boxes, max_persons=MAX_PERSONS)  # 중복 박스 병합
        except Exception as e:                        # 추론 오류 시
            print(f"⚠️ YOLO 오류: {e}")

        with frame_lock:                              # 전역 상태 갱신 (동기화)
            latest_vis = vis                          # 표시용 프레임
            latest_level = level                      # 상대온도 지수
            latest_boxes = boxes                      # 탐지 박스


# ============================================================
# 실시간 MJPEG 스트림 (YOLO 박스 표시)
# ============================================================
def generate_stream():
    """YOLO 박스를 그린 영상을 MJPEG 스트림으로 생성한다."""
    while True:                                       # 무한 루프
        try:
            with frame_lock:                          # 최신 프레임 복사
                vis = latest_vis.copy() if latest_vis is not None else None
                boxes = list(latest_boxes)            # 박스 복사
            if vis is None:                           # 프레임 없으면 대기 화면
                ph = np.zeros((480, 640, 3), np.uint8)  # 검은 화면
                cv2.putText(ph, "Waiting for thermal...", (60, 240),  # 대기 메시지
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
                _, jpeg = cv2.imencode(".jpg", ph)    # JPEG 인코딩
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                time.sleep(0.1)                       # 대기
                continue

            ann = vis.copy()                          # 표시용 복사본
            for x1, y1, x2, y2, cf in boxes:          # 각 박스 그리기
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록 박스
                cv2.putText(ann, f"person {cf:.2f}", (x1, y1 - 6),  # 신뢰도 표시
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(ann, f"YOLO: {len(boxes)} person", (10, 25),  # 탐지 인원 표시
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            _, jpeg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])  # 인코딩
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        except Exception as e:                        # 스트림 오류 시
            print(f"⚠️ 스트림 오류: {e}")
            time.sleep(0.1)
        time.sleep(0.04)                              # 약 25fps 제한


# ============================================================
# 판별 시각화: 3등분 선 + T/M/B 지수 그리기
# ============================================================
def split_mask_by_boxes(full_mask, boxes):
    """텍스트 분할로 얻은 전체 사람 마스크를 각 YOLO 박스 영역으로 나눈다.

    텍스트 프롬프트는 사람 전체를 한 번에 찾으므로, 여러 명이면
    각 박스 영역과 교집합을 내어 사람별 마스크로 분리한다.
    """
    h, w = full_mask.shape[:2]                        # 마스크 크기
    masks = []                                         # 박스별 마스크
    for box in boxes:                                 # 각 박스
        x1, y1, x2, y2 = [int(v) for v in box[:4]]    # 박스 좌표
        x1, x2 = max(0, x1), min(w, x2)               # x 범위 제한
        y1, y2 = max(0, y1), min(h, y2)               # y 범위 제한
        box_region = np.zeros_like(full_mask)         # 박스 영역 마스크
        box_region[y1:y2, x1:x2] = True               # 박스 안만 True
        masks.append(full_mask & box_region)          # 사람 ∩ 박스 = 그 사람
    return masks                                       # 박스별 사람 마스크


def draw_mask_outline(ann, mask):
    """SAM3 사람 윤곽 마스크의 외곽선을 영상에 그린다 (배경 제외 시각화)."""
    m = (mask.astype(np.uint8)) * 255                 # bool → 0/255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선 추출
    cv2.drawContours(ann, contours, -1, (168, 85, 247), 2)  # 보라색 윤곽선 (SAM3 표시)


def draw_zones(ann, zones, box):
    """영상에 3등분 선과 T/M/B 지수를 그린다 (검은 배경으로 가독성 확보)."""
    x1, y1, x2, y2 = box                              # 박스 좌표
    for i, z in enumerate(zones):                     # 각 구역
        cv2.line(ann, (x1, z["y2"]), (x2, z["y2"]), (255, 255, 255), 1)  # 구역 경계선
        if z["mean"] is not None:                     # 측정값 있으면
            en = ["T", "M", "B"][i]                   # 상/중/하 → 영어 (한글 깨짐 방지)
            txt = f"{en}{z['mean']}"                  # 표시 문자열
            tx, ty = x1 + 4, z["y1"] + 14             # 글자 위치 (구역 상단)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)  # 글자 크기
            cv2.rectangle(ann, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 3), (0, 0, 0), -1)  # 검은 배경
            cv2.putText(ann, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,  # 노란 글자
                        0.35, (0, 255, 255), 1)


# ============================================================
# Flask 라우트
# ============================================================
@app.after_request
def add_no_cache(response):
    """모든 응답에 캐시 금지 헤더를 추가한다 (브라우저 과거 화면 방지)."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"  # 캐시 금지
    response.headers["Pragma"] = "no-cache"          # HTTP 1.0 캐시 금지
    response.headers["Expires"] = "0"                # 즉시 만료
    return response


@app.route("/")
def index():
    """메인 페이지를 렌더링한다."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """실시간 YOLO MJPEG 스트림 엔드포인트."""
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/run_sam3", methods=["POST"])
def run_sam3():
    """[2단계] 현재 프레임을 캡처하고 SAM3로 사람 윤곽 마스크를 딴다.

    방식: 텍스트 프롬프트("person") 우선 — 사람 열화상에서 박스보다 정확.
          텍스트 실패 시 박스 프롬프트(percentile)로 폴백.
    """
    body = request.get_json(silent=True) or {}        # 요청 본문
    pct = float(body.get("percentile", 92))           # 박스 폴백용 백분위수
    use_text = body.get("use_text", True)             # 텍스트 방식 사용 여부 (기본 ON)
    with frame_lock:                                  # 현재 프레임 캡처 (이후 단계 공유)
        vis = latest_vis.copy() if latest_vis is not None else None
        level = latest_level.copy() if latest_level is not None else None
        boxes = list(latest_boxes)
    if vis is None or level is None:                  # 프레임 없으면 오류
        return jsonify({"ok": False, "error": "카메라 프레임 없음 (flirone 확인)"})
    if not boxes:                                     # 박스 없으면 오류
        return jsonify({"ok": False, "error": "탐지된 사람 없음 — 카메라 앞에 서세요"})

    masks = None                                       # 마스크 초기화
    method = ""                                        # 사용한 방식 (UI 표시용)
    if SAM3_ENABLED:                                   # SAM3 사용 설정이면
        if use_text:                                  # 텍스트 방식 우선
            # SAM3 "person" 텍스트로 사람 전체 윤곽 1개 받기
            full_mask = request_person_mask_text(vis, "person")  # 의미 기반 분할
            if full_mask is not None:                 # 텍스트 분할 성공
                # 전체 사람 마스크를 각 YOLO 박스 영역으로 잘라 박스별 마스크 생성
                masks = split_mask_by_boxes(full_mask, boxes)  # 박스별 분할
                method = "텍스트(person)"             # 방식 기록
        if masks is None:                             # 텍스트 실패 또는 미사용
            masks = request_person_masks(vis, boxes, percentile=pct)  # 박스 분할 폴백
            method = f"박스(pct={pct:.0f})"           # 방식 기록
    if masks is None:                                 # SAM3 전체 실패
        return jsonify({"ok": False, "error": "SAM3 윤곽 실패 (4090 서버 확인)"})

    # 윤곽선을 영상에 그려 확인용 이미지 생성
    ann = vis.copy()                                  # 표시용 복사본 (원본 열화상 유지)
    for x1, y1, x2, y2, cf in boxes:                  # YOLO 박스도 함께 표시
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 초록 박스(얇게)
    for m in masks:                                   # 각 윤곽
        draw_mask_outline(ann, m)                     # 보라색 윤곽선

    # 다음 단계(세분화/판별)가 쓸 수 있게 저장
    with stage_lock:
        stage_state.update({"vis": vis, "level": level, "boxes": boxes,
                            "masks": masks, "zones": None})
    areas = [round(float(m.sum()) / m.size * 100, 1) for m in masks]  # 각 윤곽 면적%
    print(f"🎯 SAM3 윤곽 완료 [{method}]: 면적 {areas}%")  # 진단 로그
    return jsonify({"ok": True, "image": to_b64(ann),  # 윤곽 확인 이미지
                    "areas": areas, "method": method})


@app.route("/run_segment", methods=["POST"])
def run_segment():
    """[3단계] SAM3 윤곽 안에서 3등분 세분화 측정을 수행한다."""
    with stage_lock:                                  # SAM3 단계 결과 가져오기
        vis = stage_state["vis"].copy() if stage_state["vis"] is not None else None
        level = stage_state["level"].copy() if stage_state["level"] is not None else None
        boxes = list(stage_state["boxes"])
        masks = stage_state["masks"]
    if vis is None or masks is None:                  # SAM3 먼저 안 했으면
        return jsonify({"ok": False, "error": "SAM3 윤곽을 먼저 실행하세요"})

    ann = vis.copy()                                  # 표시용 복사본 (원본 열화상 유지)
    all_zones = []                                     # 전체 구역 결과
    for idx, box in enumerate(boxes):                 # 각 사람
        zones, clean_box = measure_zones_mask(level, box, masks[idx], ZONE_NAMES)  # 윤곽 안 3등분
        draw_mask_outline(ann, masks[idx])            # 윤곽선
        draw_zones(ann, zones, clean_box)             # 3등분 선 + 지수
        all_zones.append(zones)                       # 결과 누적
        print(f"📐 세분화 {clean_box}: " +             # 진단 로그
              ", ".join(f"{z['name']}={z['mean']}" for z in zones))

    with stage_lock:                                  # 측정 결과 저장 (판별 단계용)
        stage_state["zones"] = all_zones
    return jsonify({"ok": True, "image": to_b64(ann),  # 세분화 결과 이미지
                    "zones": all_zones, "alert_level": ALERT_LEVEL})


@app.route("/run_judge", methods=["POST"])
def run_judge():
    """[4단계] 세분화 결과 영상으로 InternVL 판정을 수행한다."""
    with stage_lock:                                  # 세분화 단계 결과 가져오기
        vis = stage_state["vis"].copy() if stage_state["vis"] is not None else None
        level = stage_state["level"].copy() if stage_state["level"] is not None else None
        boxes = list(stage_state["boxes"])
        masks = stage_state["masks"]
        all_zones = stage_state["zones"]
    if vis is None or all_zones is None:              # 세분화 먼저 안 했으면
        return jsonify({"ok": False, "error": "세분화 측정을 먼저 실행하세요"})

    # 판정용 영상 만들기 (윤곽 + 3등분 표시)
    # 화면 표시용: 원본 열화상 + 윤곽선 + 3등분 지수
    ann = vis.copy()                                  # 표시용 복사본 (원본 열화상 유지)
    for idx, box in enumerate(boxes):                 # 각 사람
        zones, clean_box = measure_zones_mask(level, box, masks[idx], ZONE_NAMES)  # 측정 재현
        draw_mask_outline(ann, masks[idx])            # 윤곽선
        draw_zones(ann, zones, clean_box)             # 3등분 표시

    # InternVL 판정 (192.168.0.36) — 원본 열화상으로 판정
    first_zones = all_zones[0] if all_zones else None  # 첫 대상 측정값 (VLM 힌트)
    vlm_res = judge_with_vlm(ann, zones_info=first_zones)  # VLM 판정 요청
    analysis = format_verdict(vlm_res)                # 판정 텍스트

    with btn_lock:                                    # 결과 저장
        judge_result.update({"zones": all_zones, "analysis": analysis,
                            "image": to_b64(ann), "ts": time.time()})
    return jsonify({"ok": True, "image": judge_result["image"],  # 판정 결과
                    "zones": all_zones, "analysis": analysis,
                    "alert_level": ALERT_LEVEL})


@app.route("/status")
def status():
    """서버 상태를 반환한다."""
    with frame_lock:                                  # 현재 탐지 인원
        n = len(latest_boxes)
    return jsonify({"yolo_loaded": yolo_model is not None,  # YOLO 로드 여부
                    "thermal_device": DEVICE_INDEX,   # 카메라 번호
                    "live_boxes": n,                  # 실시간 탐지 인원
                    "grayscale_input": GRAY_INPUT,    # 흑백 입력 여부
                    "sam3_online": check_sam3_online() if SAM3_ENABLED else False,  # 4090 SAM3 연결
                    "alert_level": ALERT_LEVEL})      # 경고 기준


# ============================================================
# 서버 시작
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🌡️ 열화상 3등분 온도측정 AI [v2026-06-23 InternVL 판정판]")
    print("   YOLO 실시간 + 상대지수 재정규화 + InternVL 이미지 판정")
    print("=" * 60)
    init_yolo()                                       # YOLO 모델 로드
    threading.Thread(target=capture_thread, daemon=True).start()  # 카메라 스레드 시작
    print(f"\n🌐 웹서버: http://{FLASK_HOST}:{FLASK_PORT}")
    print("=" * 60)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)  # 서버 실행
