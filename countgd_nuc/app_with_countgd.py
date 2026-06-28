# 재고조사 Flask 서버 — 누크 웹카메라로 겹친 컵을 GDINO→SAM3→crop→VLM 파이프라인으로 세고 JSON에 저장
# [2026-06-28 작성] 컵 스택 rim 카운팅 재고조사 실습 (Stage 1)

import os
import io
import cv2
import json
import time
import base64
import threading
import datetime

import numpy as np
import requests
from flask import Flask, render_template, Response, jsonify, request

# ============================================================
# ★ 사용자 환경에 맞게 수정할 설정값들
# ============================================================
CAM_INDEX = 0                       # ← 누크 웹카메라 장치 번호 (/dev/video0)
CAM_WIDTH = 1280                    # ← 카메라 해상도 가로 (rim 세밀 카운팅 위해 고해상도 권장)
CAM_HEIGHT = 720                    # ← 카메라 해상도 세로
FLASK_HOST = "0.0.0.0"             # ← 모든 네트워크에서 접속 허용
FLASK_PORT = 5000                   # ← Flask 웹서버 포트

# 4090 PC의 AI 서버 주소 (빅맨님 표준 포트 구성)
GPU_IP = "192.168.0.75"            # ← 4090 PC IP
GDINO_URL = f"http://{GPU_IP}:5004/detect"   # Grounding DINO (객체 규정)
SAM3_URL = f"http://{GPU_IP}:5001/segment"   # SAM3 (세그멘테이션)
VLM_URL = f"http://{GPU_IP}:5003/chat"       # VLM (rim 세밀 카운팅)
COUNTGD_URL = f"http://{GPU_IP}:5005/count"  # CountGD (텍스트 프롬프트 카운팅)

GDINO_PROMPT = "cup. stacked cups."  # 컵을 규정하는 텍스트 프롬프트 (마침표 구분)
INVENTORY_JSON = "inventory.json"    # 재고 기록 저장 파일

# 누크 로컬 YOLO (겹침 영역 재탐지용) — OpenVINO 가속
YOLO_MODEL_PATH = "/home/zeta/lerobot/yolo26n_openvino_model/"  # ← YOLO OpenVINO 절대경로
YOLO_IMGSZ = 640                     # ← OpenVINO 변환 크기와 반드시 일치!
USE_YOLO_REFINE = True               # ← True면 겹침 영역에 YOLO 재탐지 융합
USE_COUNTGD = True                   # ← True면 겹침 영역에 CountGD 카운팅 융합 (4090 서버 필요)
COUNTGD_PROMPT = "cup"               # ← CountGD 카운팅 대상 텍스트

# LVLM-Count 분할정복 카운팅 (대량/밀집 객체용)
USE_DIVIDE_COUNT = True              # ← True면 큰 crop을 격자 분할해 세밀 카운팅
DIVIDE_TRIGGER_COUNT = 15            # ← 1차 카운트가 이 값 이상이면 분할 카운팅 발동
DIVIDE_GRID = 2                      # ← 분할 격자 크기 (2 = 2x2 = 4타일)
DIVIDE_OVERLAP = 0.15               # ← 타일 겹침 비율 (경계 객체 누락 방지)

# ============================================================
# 전역 상태 (스레드 간 공유)
# ============================================================
app = Flask(__name__)

latest_frame = None                       # 웹카메라 최신 프레임 (BGR numpy)
latest_frame_lock = threading.Lock()      # 프레임 접근 동기화
yolo_model = None                         # 누크 로컬 YOLO 모델 (재탐지용)


# ============================================================
# 1. 웹카메라 캡처 스레드
# ============================================================
def camera_capture_thread():
    """누크 웹카메라에서 계속 프레임을 읽어 전역 변수에 갱신하는 스레드."""
    global latest_frame

    cap = cv2.VideoCapture(CAM_INDEX)                         # 카메라 열기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)             # 해상도 가로 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)          # 해상도 세로 설정
    print(f"📷 카메라 시작: index={CAM_INDEX}, {CAM_WIDTH}x{CAM_HEIGHT}")

    while True:
        ok, frame = cap.read()                               # 프레임 한 장 읽기
        if ok:
            with latest_frame_lock:
                latest_frame = frame                         # 최신 프레임 갱신
        else:
            time.sleep(0.1)                                  # 읽기 실패 시 잠시 대기


# ============================================================
# 2. AI 서버 호출 헬퍼 (이미지를 base64로 보내고 결과 받기)
# ============================================================
def _encode_b64(image_bgr):
    """BGR 이미지를 JPEG base64 문자열로 인코딩한다."""
    _, jpeg = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])  # 고품질 JPEG
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')                     # base64 변환


def call_gdino(image_bgr, prompt):
    """Grounding DINO 서버에 이미지를 보내 컵 영역 박스를 받는다."""
    payload = {"image": _encode_b64(image_bgr), "text": prompt}  # 요청 본문
    r = requests.post(GDINO_URL, json=payload, timeout=30)       # GDINO 호출
    return r.json().get("detections", [])                        # [{box, score, label}, ...]


def call_sam3(image_bgr, boxes):
    """SAM3 서버에 박스를 보내 각 컵/스택의 마스크를 받는다."""
    payload = {"image": _encode_b64(image_bgr), "boxes": boxes}  # 박스 프롬프트
    r = requests.post(SAM3_URL, json=payload, timeout=60)        # SAM3 호출
    return r.json().get("masks", [])                            # base64 PNG 마스크 리스트


def yolo_recount(crop_bgr):
    """확대한 겹침 영역 crop에 YOLO를 재추론하여 컵을 다시 센다 (결정론적 보정).

    원본 전체에서는 작아서 놓친 컵을, 확대 화면에서 다시 탐지한다.
    rim 스택보다 '좌우로 흩어진 겹침'에서 효과적 (몸통이 보이므로).
    반환: (개수, 평균신뢰도)
    """
    global yolo_model
    if yolo_model is None:
        return 0, 0.0

    # COCO 기준 'cup' 클래스 id=41 만 카운팅
    results = yolo_model(crop_bgr, verbose=False, imgsz=YOLO_IMGSZ, conf=0.30)  # YOLO 재추론
    boxes = results[0].boxes                                          # 탐지 박스들
    if boxes is None or len(boxes) == 0:
        return 0, 0.0

    cls = boxes.cls.cpu().numpy().astype(int)                        # 클래스 id 배열
    conf = boxes.conf.cpu().numpy()                                  # 신뢰도 배열
    cup_mask = (cls == 41)                                           # cup 클래스만 선택
    n = int(cup_mask.sum())                                          # 컵 개수
    avg_conf = float(conf[cup_mask].mean()) if n > 0 else 0.0        # 평균 신뢰도
    return n, avg_conf


def countgd_recount(crop_bgr, prompt=COUNTGD_PROMPT):
    """확대한 겹침 영역 crop을 CountGD 서버(4090:5005)에 보내 카운팅한다.

    CountGD는 Grounding DINO 기반 카운팅 전용 모델로, 자기유사 객체
    이중 카운팅 방지가 내장되어 겹친 컵에 강하다.
    반환: (개수, 평균신뢰도, 박스리스트[cx,cy,w,h 정규화])
    """
    try:
        payload = {"image": _encode_b64(crop_bgr), "text": prompt}   # 카운팅 요청
        r = requests.post(COUNTGD_URL, json=payload, timeout=30)     # CountGD 호출
        data = r.json()
        scores = data.get("scores", [])                              # 박스별 신뢰도
        boxes = data.get("boxes", [])                                # 박스 좌표 (정규화)
        n = int(data.get("count", 0))                                # 카운트
        avg_conf = float(sum(scores) / len(scores)) if scores else 0.0  # 평균 신뢰도
        return n, avg_conf, boxes
    except Exception as e:
        print(f"⚠️ CountGD 호출 실패: {e}")
        return 0, 0.0, []


def divide_and_count(crop_bgr, count_fn, grid=DIVIDE_GRID, overlap=DIVIDE_OVERLAP):
    """LVLM-Count 분할정복 카운팅 — 큰 crop을 격자로 쪼개 세밀하게 센다.

    핵심: 타일을 약간 겹치게 자르되, 객체 중심이 그 타일의 '담당 영역(코어)'에
    들어올 때만 카운팅한다. 경계에 걸친 객체의 이중 카운팅을 원천 차단한다.

    Args:
        crop_bgr: 확대된 겹침 영역
        count_fn: 타일을 받아 (개수, 신뢰도, 박스리스트) 반환하는 함수 (CountGD)
        grid: 격자 크기 (2 → 2x2)
        overlap: 타일 간 겹침 비율 (경계 객체 누락 방지)
    Returns:
        (총개수, 분할정보dict)
    """
    H, W = crop_bgr.shape[:2]
    tile_h, tile_w = H // grid, W // grid                            # 코어 타일 기본 크기
    pad_h, pad_w = int(tile_h * overlap), int(tile_w * overlap)      # 겹침 여백
    total = 0
    tile_counts = []

    for r in range(grid):
        for c in range(grid):
            # 코어 영역 (이 타일이 '담당'하는 중복 없는 구역)
            core_x1, core_y1 = c * tile_w, r * tile_h               # 코어 좌상단
            core_x2, core_y2 = core_x1 + tile_w, core_y1 + tile_h   # 코어 우하단

            # 실제 자를 영역 (코어 + 겹침 여백 → 경계 객체도 온전히 보이게)
            cut_x1 = max(0, core_x1 - pad_w)                        # 여백 포함 좌상단
            cut_y1 = max(0, core_y1 - pad_h)
            cut_x2 = min(W, core_x2 + pad_w)                        # 여백 포함 우하단
            cut_y2 = min(H, core_y2 + pad_h)
            tile = crop_bgr[cut_y1:cut_y2, cut_x1:cut_x2]           # 타일 잘라내기

            n, conf, boxes = count_fn(tile)                        # 타일 카운팅
            th, tw = tile.shape[:2]

            # 경계 중복 제거: 객체 '중심'이 코어 영역에 있을 때만 카운팅
            kept = 0
            for box in boxes:                                      # box = [cx,cy,w,h] 정규화
                # 타일 내 정규화 좌표 → 전체 crop 픽셀 좌표로 환산
                obj_cx = cut_x1 + box[0] * tw                      # 객체 중심 x (crop 기준)
                obj_cy = cut_y1 + box[1] * th                      # 객체 중심 y (crop 기준)
                # 중심이 이 타일의 코어 영역 안에 있으면 카운팅
                if core_x1 <= obj_cx < core_x2 and core_y1 <= obj_cy < core_y2:
                    kept += 1                                      # 담당 객체로 인정
            # CountGD가 박스를 안 줄 경우(좌표 없음) 폴백: 단순 개수 사용
            tile_count = kept if boxes else n
            total += tile_count
            tile_counts.append(tile_count)

    return total, {"grid": grid, "tiles": tile_counts, "overlap": overlap}


def call_vlm_count(crop_bgr):
    """확대한 겹침 영역 crop을 VLM에 보내 rim(테두리) 개수를 센다."""
    # 베테랑 재고조사관 역할 부여 (빅맨님 열화상/검사 시스템과 동일 패턴)
    prompt = (
        "당신은 20년 경력의 물류창고 재고조사관입니다. "
        "이 사진은 컵을 옆에서 본 모습이며, 컵들이 포개져 쌓여 있습니다. "
        "컵의 테두리(rim) 줄무늬를 하나씩 세어 총 몇 개가 쌓여 있는지 판단하세요. "
        "겹쳐서 애매하면 가장 신뢰할 수 있는 개수를 제시하세요. "
        "반드시 다음 JSON 형식으로만 답하세요: "
        '{"count": 숫자, "reason": "판단 근거 한 줄"}'
    )
    payload = {"image": _encode_b64(crop_bgr), "prompt": prompt}  # VLM 요청
    r = requests.post(VLM_URL, json=payload, timeout=60)         # VLM 호출
    text = r.json().get("response", "{}")                        # VLM 응답 텍스트

    # VLM이 JSON 외 텍스트를 섞어 보낼 수 있으니 중괄호 구간만 추출
    try:
        s, e = text.find('{'), text.rfind('}') + 1              # JSON 시작/끝 위치
        parsed = json.loads(text[s:e])                          # JSON 파싱
        return int(parsed.get("count", 0)), parsed.get("reason", "")
    except Exception:
        return 0, f"파싱 실패: {text[:80]}"                     # 실패 시 0개 처리


# ============================================================
# 3. 핵심 파이프라인: 겹친 컵 카운팅
# ============================================================
def decode_mask(mask_b64, h, w):
    """base64 PNG 마스크를 numpy 이진 배열로 디코딩한다."""
    raw = base64.b64decode(mask_b64)                                  # base64 디코딩
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)  # PNG → 그레이
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h))                                # 원본 크기로 정렬
    return (arr > 127).astype(np.uint8)                              # 이진 마스크 (0/1)


def analyze_occlusion(mask, all_boxes, my_idx):
    """SAM3 마스크의 기하 분석으로 '겹침/스택 여부'를 판단한다 (모델 없이).

    겹침 신호 3가지:
      - 박스 IoU 높음   → 다른 컵과 포개짐
      - 종횡비 길쭉함    → 컵이 일렬로 쌓인 스택
      - convexity 낮음   → 윤곽이 가려져 움푹 들어감
    """
    ys, xs = np.where(mask > 0)                                       # 마스크 픽셀 좌표
    if len(xs) == 0:
        return {"is_occluded": False, "stack_likely": False, "signals": {}}

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()          # 마스크 바운딩 박스
    bw, bh = (x2 - x1 + 1), (y2 - y1 + 1)                            # 박스 너비/높이

    # [신호1] 다른 박스와의 최대 IoU (겹침 정도)
    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])                  # 교집합 좌상단
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])                  # 교집합 우하단
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)                # 교집합 폭/높이
        inter = iw * ih                                              # 교집합 면적
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter  # 합집합 면적
        return inter / ua if ua > 0 else 0.0

    my_box = [x1, y1, x2, y2]
    max_iou = 0.0
    for j, b in enumerate(all_boxes):
        if j != my_idx:
            max_iou = max(max_iou, iou(my_box, b))                  # 가장 많이 겹치는 박스

    # [신호2] 종횡비 (세로로 길면 스택 의심)
    aspect = bh / bw if bw > 0 else 0                                # 높이/너비 비율

    # [신호3] convexity = 마스크 면적 / convex hull 면적 (낮을수록 가려짐)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convexity = 1.0
    if contours:
        c = max(contours, key=cv2.contourArea)                      # 가장 큰 윤곽
        area = cv2.contourArea(c)                                   # 실제 면적
        hull_area = cv2.contourArea(cv2.convexHull(c))              # 볼록 외곽 면적
        convexity = area / hull_area if hull_area > 0 else 1.0      # 볼록성 비율

    # 종합 판정 (임계값은 현장에서 튜닝)
    is_occluded = (max_iou > 0.15) or (convexity < 0.85)            # 겹침 의심
    stack_likely = aspect > 1.4                                     # 일렬 스택 의심

    return {
        "is_occluded": bool(is_occluded),
        "stack_likely": bool(stack_likely),
        "signals": {
            "max_iou": round(float(max_iou), 3),                    # 최대 겹침도
            "aspect": round(float(aspect), 2),                      # 종횡비
            "convexity": round(float(convexity), 3)                 # 볼록성
        }
    }


def crop_by_mask(image_bgr, mask, pad=20, upscale=2.0):
    """마스크 영역을 잘라내고 확대한다 (겹침 부분 세밀 관찰용)."""
    ys, xs = np.where(mask > 0)                                      # 마스크가 있는 픽셀 좌표
    if len(xs) == 0:
        return None
    x1, x2 = max(0, xs.min() - pad), min(image_bgr.shape[1], xs.max() + pad)  # 좌우 여백
    y1, y2 = max(0, ys.min() - pad), min(image_bgr.shape[0], ys.max() + pad)  # 상하 여백
    crop = image_bgr[y1:y2, x1:x2]                                   # 영역 잘라내기
    # 보간 확대 — 겹친 rim 줄무늬를 VLM이 잘 보도록 키움
    crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return crop, (x1, y1, x2, y2)                                    # 확대 crop + 원본 좌표


def run_inventory_pipeline(frame_bgr):
    """전체 파이프라인 실행: GDINO → SAM3 → crop 확대 → VLM 카운팅."""
    h, w = frame_bgr.shape[:2]
    result = {"stacks": [], "total": 0, "annotated": None}          # 결과 누적용

    # [1] GDINO: 컵 영역 규정
    detections = call_gdino(frame_bgr, GDINO_PROMPT)               # 컵 박스 탐지
    if not detections:
        result["annotated"] = _encode_b64(frame_bgr)              # 탐지 없으면 원본 반환
        return result

    boxes = [d["box"] for d in detections]                         # 박스 좌표만 추출

    # [2] SAM3: 각 컵/스택 세그멘테이션
    masks_b64 = call_sam3(frame_bgr, boxes)                        # 마스크 리스트 받기

    annotated = frame_bgr.copy()                                  # 시각화용 복사본
    total = 0

    # [3] 각 스택마다 겹침 분석 → crop 확대 → YOLO/VLM 융합 카운팅
    for i, mask_b64 in enumerate(masks_b64):
        mask = decode_mask(mask_b64, h, w)                         # 마스크 디코딩

        # 겹침 판단 (기하 분석) — 겹쳤으면 더 크게 확대해서 본다
        occ = analyze_occlusion(mask, boxes, i)                    # 겹침/스택 여부
        zoom = 2.5 if occ["is_occluded"] or occ["stack_likely"] else 2.0  # 겹치면 확대 강화

        cropped = crop_by_mask(frame_bgr, mask, upscale=zoom)      # 겹침 영역 확대 crop
        if cropped is None:
            continue
        crop_img, (x1, y1, x2, y2) = cropped

        # ─── 융합 카운팅: 겹침 종류에 따라 CountGD/YOLO/VLM 역할 분담 ───
        # 흩어진 겹침(IoU↑) → CountGD 카운팅이 강점 (자기유사 이중카운팅 방지)
        # 일렬 스택(종횡비↑) → VLM rim 카운팅이 강점 (몸통 가려짐, 맥락 추론)
        # 둘 다/애매        → CountGD·VLM 교차검증 후 채택
        cgd_n, cgd_conf = 0, 0.0                                   # CountGD 결과
        cgd_n, cgd_conf = 0, 0.0                                   # CountGD 결과
        yolo_n, yolo_conf, vlm_n = 0, 0.0, 0                       # YOLO/VLM 결과
        divide_info = None                                        # 분할 카운팅 정보
        vlm_reason = ""
        method = ""

        if occ["is_occluded"] and not occ["stack_likely"]:
            # 케이스 A: 흩어진 겹침 → CountGD 우선 (없으면 YOLO 폴백)
            if USE_COUNTGD:
                cgd_n, cgd_conf, cgd_boxes = countgd_recount(crop_img)  # 1차 CountGD 카운팅
                count = cgd_n
                method = "countgd"
                vlm_reason = f"CountGD {cgd_n}개 (신뢰도 {cgd_conf:.2f})"

                # LVLM-Count: 객체가 많으면(밀집) 분할정복으로 재정밀화
                if USE_DIVIDE_COUNT and cgd_n >= DIVIDE_TRIGGER_COUNT:
                    # 박스 좌표만 반환하는 람다로 CountGD를 분할 카운팅에 주입
                    div_n, divide_info = divide_and_count(
                        crop_img,
                        lambda t: countgd_recount(t)               # 타일별 CountGD
                    )
                    count = div_n                                  # 분할 결과로 갱신
                    method = "countgd+divide"
                    vlm_reason = (f"분할정복: 1차 {cgd_n}개 → "
                                  f"{divide_info['grid']}x{divide_info['grid']} 분할 후 {div_n}개 "
                                  f"(타일별 {divide_info['tiles']})")
            elif USE_YOLO_REFINE:
                yolo_n, yolo_conf = yolo_recount(crop_img)         # YOLO 폴백
                count = yolo_n
                method = "yolo"
                vlm_reason = f"YOLO 재탐지 {yolo_n}개 (신뢰도 {yolo_conf:.2f})"
            else:
                vlm_n, vlm_reason = call_vlm_count(crop_img)       # VLM 폴백
                count = vlm_n
                method = "vlm"

        elif occ["stack_likely"] and not occ["is_occluded"]:
            # 케이스 B: 일렬 스택 → VLM 우선 (rim 카운팅은 VLM이 강점)
            vlm_n, vlm_reason = call_vlm_count(crop_img)           # VLM rim 카운팅
            count = vlm_n
            method = "vlm"

        else:
            # 케이스 C: 겹침+스택 둘 다 or 애매 → CountGD·VLM 교차검증
            vlm_n, vlm_reason = call_vlm_count(crop_img)           # VLM 카운팅
            if USE_COUNTGD:
                cgd_n, cgd_conf, _ = countgd_recount(crop_img)     # CountGD 카운팅
            count = max(cgd_n, vlm_n)                              # 더 많이 센 쪽 채택 (누락 방지)
            method = "fusion"
            if cgd_n != vlm_n:
                vlm_reason = f"교차검증: CountGD={cgd_n}, VLM={vlm_n} → {count} 채택. {vlm_reason}"

        total += count                                            # 전체 합산

        # 시각화: 겹침이면 빨강, 단일이면 보라
        color = (0, 80, 230) if occ["is_occluded"] else (124, 58, 237)
        tag = "OVERLAP" if occ["is_occluded"] else ("STACK" if occ["stack_likely"] else "single")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, f"#{i+1}: {count}EA [{tag}/{method}]",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        result["stacks"].append({                                 # 스택별 결과 기록
            "stack_id": i + 1,
            "count": count,
            "method": method,                                     # 카운팅 방법 (yolo/vlm/fusion)
            "yolo_count": yolo_n,                                 # YOLO 결과
            "countgd_count": cgd_n,                               # CountGD 결과
            "divide_info": divide_info,                           # 분할정복 정보 (있으면)
            "vlm_count": vlm_n,                                   # VLM 결과
            "reason": vlm_reason,
            "box": [x1, y1, x2, y2],
            "occluded": occ["is_occluded"],                       # 겹침 여부
            "stack_likely": occ["stack_likely"],                  # 일렬 스택 여부
            "signals": occ["signals"]                             # 판단 근거 수치
        })

    # 전체 개수 오버레이
    cv2.putText(annotated, f"TOTAL: {total} cups",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 200, 90), 3)

    result["total"] = total
    result["annotated"] = _encode_b64(annotated)                  # 결과 이미지 base64
    return result


# ============================================================
# 4. 재고 JSON 저장/조회
# ============================================================
def save_inventory(record):
    """재고조사 결과 1건을 inventory.json에 누적 저장한다."""
    data = []
    if os.path.exists(INVENTORY_JSON):                            # 기존 파일이 있으면 읽기
        try:
            with open(INVENTORY_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []                                             # 손상 시 빈 리스트로 시작
    data.append(record)                                          # 새 기록 추가
    with open(INVENTORY_JSON, "w", encoding="utf-8") as f:        # 전체 다시 저장
        json.dump(data, f, ensure_ascii=False, indent=2)
    return len(data)                                             # 총 기록 수 반환


def load_inventory():
    """저장된 모든 재고 기록을 불러온다."""
    if not os.path.exists(INVENTORY_JSON):
        return []
    with open(INVENTORY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 5. Flask 라우트
# ============================================================
@app.route('/')
def index():
    """메인 페이지 렌더링."""
    return render_template('index.html')


@app.route('/mobile')
def mobile():
    """핸드폰 촬영 전용 페이지 (현장에서 폰으로 접속)."""
    return render_template('mobile.html')


@app.route('/video_feed')
def video_feed():
    """웹카메라 실시간 MJPEG 스트림 (촬영 위치 잡기용)."""
    def gen():
        while True:
            with latest_frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None:
                frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)  # 대기 화면
                cv2.putText(frame, "Waiting camera...", (100, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)                                     # 약 20fps 제한
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/count', methods=['POST'])
def count():
    """현재 프레임으로 재고조사 파이프라인을 실행한다 (촬영 버튼)."""
    with latest_frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "카메라 프레임 없음"}), 503

    t0 = time.time()
    result = run_inventory_pipeline(frame)                       # 파이프라인 실행
    elapsed = round((time.time() - t0) * 1000)                   # 소요 시간(ms)

    print(f"📊 재고조사 완료: 총 {result['total']}개, {elapsed}ms")
    return jsonify({
        "total": result["total"],
        "stacks": result["stacks"],
        "image": result["annotated"],
        "elapsed_ms": elapsed
    })


@app.route('/save', methods=['POST'])
def save():
    """확인한 재고조사 결과를 서버 JSON에 저장한다 (저장 버튼)."""
    data = request.get_json()                                   # 클라이언트 결과 받기
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),  # 조사 시각
        "location": data.get("location", "미지정"),            # 창고 위치 라벨
        "total": data.get("total", 0),                          # 총 개수
        "stacks": data.get("stacks", [])                        # 스택별 상세
    }
    n = save_inventory(record)                                  # JSON에 누적 저장
    print(f"💾 재고 저장 완료 (총 {n}건 기록)")
    return jsonify({"saved": True, "record_count": n, "record": record})


@app.route('/history')
def history():
    """저장된 재고조사 기록 전체를 반환한다."""
    return jsonify({"records": load_inventory()})


# ============================================================
# 5-2. 핸드폰 촬영 업로드 (현장에서 폰으로 찍어 바로 조사)
# ============================================================
@app.route('/upload', methods=['POST'])
def upload():
    """핸드폰으로 찍은 사진을 받아 재고조사 파이프라인을 실행한다.

    Request: multipart/form-data (file=사진) 또는 JSON {"image": base64}
    핸드폰 브라우저의 카메라 캡처(<input capture>)로 바로 업로드된다.
    """
    frame = None

    # [방법1] 파일 업로드 (multipart)
    if 'file' in request.files:
        file_bytes = request.files['file'].read()                       # 업로드 바이트
        frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    # [방법2] base64 JSON (앱/자바스크립트 캡처)
    elif request.is_json and request.get_json().get("image"):
        raw = base64.b64decode(request.get_json()["image"])             # base64 디코딩
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "이미지를 읽을 수 없음"}), 400

    t0 = time.time()
    result = run_inventory_pipeline(frame)                              # 파이프라인 실행
    elapsed = round((time.time() - t0) * 1000)

    print(f"📱 핸드폰 업로드 조사 완료: 총 {result['total']}개, {elapsed}ms")
    return jsonify({
        "total": result["total"],
        "stacks": result["stacks"],
        "image": result["annotated"],
        "elapsed_ms": elapsed
    })


# ============================================================
# 5-3. 재고조사 보고서 (HTML / 요약 통계)
# ============================================================
@app.route('/report')
def report():
    """저장된 모든 기록을 종합한 재고조사 보고서 페이지를 렌더링한다."""
    records = load_inventory()                                          # 전체 기록

    # 요약 통계 계산
    total_surveys = len(records)                                       # 조사 횟수
    grand_total = sum(r.get("total", 0) for r in records)              # 누적 컵 수
    occluded_cnt = sum(
        1 for r in records for s in r.get("stacks", []) if s.get("occluded")
    )                                                                  # 겹침 스택 수
    # 위치별 최신 재고 집계 (같은 위치는 가장 최근 조사값 사용)
    by_location = {}
    for r in records:
        by_location[r.get("location", "미지정")] = r                   # 최신값으로 덮어씀

    summary = {
        "total_surveys": total_surveys,                                # 총 조사 횟수
        "grand_total": grand_total,                                    # 누적 컵 수
        "occluded_count": occluded_cnt,                                # 겹침 감지 횟수
        "locations": len(by_location)                                  # 조사 위치 수
    }
    return render_template('report.html', records=records,
                           summary=summary, by_location=by_location)


@app.route('/report/data')
def report_data():
    """보고서용 원본 데이터를 JSON으로 반환한다 (차트/외부 연동용)."""
    return jsonify({"records": load_inventory()})


# ============================================================
# 6. 서버 시작
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("📦 컵 재고조사 시스템 — GDINO→SAM3→기하분석→YOLO/VLM 융합")
    print("   [v2026-06-28 YOLO 융합판]")
    print("=" * 60)

    # 누크 로컬 YOLO 로드 (겹침 영역 재탐지용)
    if USE_YOLO_REFINE:
        try:
            from ultralytics import YOLO                          # YOLO 임포트 (선택적)
            print(f"⚡ YOLO 재탐지 모델 로딩: {YOLO_MODEL_PATH}")
            yolo_model = YOLO(YOLO_MODEL_PATH, task="detect")     # OpenVINO 모델 로드
            print("✅ YOLO 융합 활성화")
        except Exception as e:
            print(f"⚠️ YOLO 로드 실패 — VLM 단독 모드로 동작: {e}")
            yolo_model = None

    # 카메라 캡처 스레드 시작 (데몬 = 메인 종료 시 함께 종료)
    cam_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    cam_thread.start()

    print(f"🌐 웹 서버 시작: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"   브라우저에서 http://누크IP:{FLASK_PORT} 으로 접속하세요")
    print("=" * 60)

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
