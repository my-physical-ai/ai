# SAM3 정밀 분할 서버 — RTX 4090에서 실행
# [2026-06-20 작성] NUC(GDINO)가 보낸 박스를 받아 SAM3 정밀 마스크를 생성해 회신
#
# ★ 역할 분담:
#   NUC  (app.py, CPU): YOLO 트래킹 + GDINO 박스 탐지 "어디에 있는지(박스)"
#   4090 (이 서버, GPU): GDINO 박스를 prompt로 받아 "정확한 픽셀 경계(마스크)" 생성
#
# ★ 통신: NUC가 HTTP POST로 {image(base64), boxes} 전송 → 이 서버가 마스크 오버레이 회신
#
# 실행:
#   4090 PC: conda activate sam3 && python sam3_server.py
#   (NUC app.py의 SAM3_SERVER_URL을 이 PC의 IP:5001로 맞출 것)
#
# ⚠️ SAM3 로딩 API는 설치 방식마다 다름 → load_sam3_model() 안에서 본인 환경에 맞게 택1

import base64
import time

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================
# ★ 설정값
# ============================================================
SAM3_CHECKPOINT = "sam3.pt"          # ← 4090에 설치된 SAM3 체크포인트 경로
DEVICE = "cuda"                       # 4090 GPU 사용 (CPU면 "cpu")
FLASK_HOST = "0.0.0.0"               # NUC가 접속할 수 있게 모든 인터페이스 허용
FLASK_PORT = 5001                     # NUC app.py의 SAM3_SERVER_URL 포트와 일치

# 전역 SAM3 객체 (방식에 따라 predictor 또는 processor)
sam3_engine = None
SAM3_BACKEND = None                   # "ultralytics" 또는 "official" — load 시 자동 설정


# ============================================================
# 1. SAM3 모델 로딩 (★ 본인 설치 방식에 맞게 택1)
# ============================================================
def load_sam3_model():
    """SAM3 모델을 로드한다. 설치 방식에 따라 아래 A 또는 B 중 하나를 사용.

    Returns:
        (engine, backend_name) 튜플
    """
    global SAM3_BACKEND

    # ─────────────────────────────────────────────
    # 방식 A: Ultralytics SAM3 (pip install ultralytics)
    #   - 박스 프롬프트: inference_features(..., bboxes=[[x1,y1,x2,y2]])
    #   - 가장 간단. ultralytics가 이미 깔려 있으면 우선 권장
    # ─────────────────────────────────────────────
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor  # SAM3 예측기
        overrides = dict(
            conf=0.25, task="segment", mode="predict",
            model=SAM3_CHECKPOINT, verbose=False,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)    # 예측기 생성
        predictor.setup_model()                                   # 모델 로드(GPU)
        SAM3_BACKEND = "ultralytics"
        print(f"✅ SAM3 로드 완료 (Ultralytics 방식, device={DEVICE})")
        return predictor, SAM3_BACKEND
    except Exception as e:
        print(f"⚠️ Ultralytics SAM3 로드 실패 → 공식 패키지 방식 시도: {e}")

    # ─────────────────────────────────────────────
    # 방식 B: 공식 sam3 패키지 (facebookresearch/sam3)
    #   - build_sam3_image_model + Sam3Processor
    #   - 박스 프롬프트는 set_box_prompt 계열 API 사용
    # ─────────────────────────────────────────────
    from sam3 import build_sam3_image_model                       # 공식 빌더
    from sam3.model.sam3_image_processor import Sam3Processor      # 공식 프로세서

    model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
    processor = Sam3Processor(model)                              # 프로세서 래핑
    SAM3_BACKEND = "official"
    print(f"✅ SAM3 로드 완료 (공식 sam3 패키지 방식, device={DEVICE})")
    return processor, SAM3_BACKEND


# ============================================================
# 2. 박스 프롬프트로 마스크 추출 (백엔드별 분기)
# ============================================================
def segment_with_boxes(frame_rgb: np.ndarray, boxes: list) -> list:
    """이미지와 박스 리스트를 받아 박스별 마스크(bool 배열)를 반환한다.

    Args:
        frame_rgb: RGB numpy 이미지 (H, W, 3)
        boxes: 박스 리스트 [[x1,y1,x2,y2], ...]

    Returns:
        마스크 리스트 [np.ndarray(H,W) bool, ...] (박스 순서대로)
    """
    masks_out = []

    if SAM3_BACKEND == "ultralytics":
        # ── Ultralytics 방식: 이미지 임베딩 1회 → 박스마다 재사용 ──
        sam3_engine.set_image(frame_rgb)                          # 임베딩 1회 계산(무거움)
        feats = sam3_engine.features                              # 추출된 특징 재사용
        h, w = frame_rgb.shape[:2]

        for box in boxes:
            # bboxes 인자로 박스 프롬프트 전달 (박스당 1마스크)
            masks, _boxes = sam3_engine.inference_features(
                feats, src_shape=(h, w), bboxes=[box]
            )
            if masks is not None and len(masks) > 0:
                m = masks[0].cpu().numpy().astype(bool)           # (H,W) bool
                masks_out.append(m)

    else:
        # ── 공식 sam3 패키지 방식 ──
        from PIL import Image
        image_pil = Image.fromarray(frame_rgb)                    # RGB → PIL
        state = sam3_engine.set_image(image_pil)                  # 임베딩 1회 계산

        for box in boxes:
            # 박스 프롬프트 설정 (API명은 버전에 따라 set_box_prompt 등으로 다를 수 있음)
            results = sam3_engine.set_box_prompt(state=state, box=box)
            # results에서 마스크 추출 (구조는 설치 버전에 맞게 조정)
            m = results["masks"][0].astype(bool) if "masks" in results else None
            if m is not None:
                masks_out.append(m)

    return masks_out


# ============================================================
# 3. Flask 라우트
# ============================================================
@app.route('/ping')
def ping():
    """헬스체크 — NUC가 /status에서 4090 연결 확인용."""
    return jsonify({"status": "ok", "backend": SAM3_BACKEND, "device": DEVICE})


@app.route('/segment', methods=['POST'])
def segment():
    """NUC에서 받은 이미지 + GDINO 박스로 SAM3 정밀 마스크를 생성하고,
    박스(사각형) vs 마스크(픽셀) 비교 결과를 반환한다.

    Request JSON:
        {"image": "base64 JPEG", "boxes": [[x1,y1,x2,y2], ...]}

    Response JSON:
        {
          "box_image": base64,        # 왼쪽: GDINO 박스를 빨강으로 채운 이미지
          "mask_image": base64,       # 오른쪽: SAM3 마스크를 초록으로 칠한 이미지
          "objects": [{박스면적, 마스크면적, 정확도%, 배경%}, ...],
          "total_box_area", "total_mask_area", "precision_pct", ...
        }
    """
    data = request.get_json()
    if not data or 'image' not in data or 'boxes' not in data:
        return jsonify({"error": "image와 boxes 필요"}), 400

    boxes = data['boxes']
    if len(boxes) == 0:
        return jsonify({"error": "박스가 비어있음"}), 400

    # ── base64 → BGR 이미지 디코딩 ──
    img_bytes = base64.b64decode(data['image'])
    frame_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)  # JPEG → BGR
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)           # SAM3는 RGB 입력

    # ── SAM3 추론 (박스 프롬프트) ──
    t0 = time.time()
    with torch.no_grad():
        masks = segment_with_boxes(frame_rgb, boxes)                # 박스별 마스크
    infer_ms = (time.time() - t0) * 1000

    # ── 비교용 이미지 2장 준비 ──
    box_view = frame_bgr.copy()      # 왼쪽: 박스(사각형) 채움
    mask_view = frame_bgr.copy()     # 오른쪽: 마스크(픽셀) 채움

    RED = np.array([60, 60, 230])    # 박스 = 빨강 (BGR) — "배경까지 포함"
    GREEN = np.array([80, 220, 90])  # 마스크 = 초록 (BGR) — "물체만 정확히"

    objects = []
    total_box_area = 0
    total_mask_area = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)

        box_area = (x2 - x1) * (y2 - y1)                            # 박스 면적(픽셀)

        # ── 왼쪽: 박스 전체를 반투명 빨강으로 채움 (사각형이라 배경 포함) ──
        roi = box_view[y1:y2, x1:x2]
        box_view[y1:y2, x1:x2] = (roi * 0.55 + RED * 0.45).astype(np.uint8)
        cv2.rectangle(box_view, (x1, y1), (x2, y2), (40, 40, 220), 2)

        # ── 오른쪽: 마스크만 반투명 초록으로 채움 + 외곽선 ──
        mask_area = 0
        if i < len(masks) and masks[i] is not None:
            mask = masks[i]
            mask_area = int(mask.sum())                              # 마스크 면적(픽셀)

            # 마스크 영역만 초록으로 (픽셀 단위)
            mask_view[mask] = (mask_view[mask] * 0.45 + GREEN * 0.55).astype(np.uint8)

            # 마스크 외곽선(컨투어)을 진한 초록으로 그려 경계 강조
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_view, contours, -1, (40, 180, 40), 2)

        # ── 정확도 계산: 박스 안에서 실제 물체(마스크)가 차지하는 비율 ──
        precision = (mask_area / box_area * 100) if box_area > 0 else 0
        background = 100 - precision                                 # 박스 중 배경 비율

        objects.append({
            "id": i + 1,
            "box_area": box_area,                                   # 박스 픽셀 수
            "mask_area": mask_area,                                 # 마스크 픽셀 수
            "precision_pct": round(precision, 1),                   # 물체 비율
            "background_pct": round(background, 1)                  # 버려진 배경 비율
        })
        total_box_area += box_area
        total_mask_area += mask_area

    # ── 각 이미지에 라벨 부착 ──
    cv2.putText(box_view, "BOX (GDINO) - rectangle", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 220), 2)
    cv2.putText(mask_view, "MASK (SAM3) - pixel-precise", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 180, 40), 2)

    # ── 전체 정확도 ──
    total_precision = (total_mask_area / total_box_area * 100) if total_box_area > 0 else 0

    # ── 두 이미지를 base64로 인코딩 ──
    _, box_jpg = cv2.imencode('.jpg', box_view, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, mask_jpg = cv2.imencode('.jpg', mask_view, [cv2.IMWRITE_JPEG_QUALITY, 90])
    box_b64 = base64.b64encode(box_jpg.tobytes()).decode('utf-8')
    mask_b64 = base64.b64encode(mask_jpg.tobytes()).decode('utf-8')

    print(f"🎯 SAM3: {len(boxes)}박스 → {len(masks)}마스크, {infer_ms:.0f}ms, "
          f"정확도 {total_precision:.0f}% (배경 {100-total_precision:.0f}% 제거)")

    return jsonify({
        "box_image": box_b64,                                      # 왼쪽 비교 이미지
        "mask_image": mask_b64,                                    # 오른쪽 비교 이미지
        "objects": objects,                                        # 물체별 면적/정확도
        "count": len(masks),
        "total_box_area": total_box_area,
        "total_mask_area": total_mask_area,
        "precision_pct": round(total_precision, 1),                # 전체 정확도
        "background_pct": round(100 - total_precision, 1),         # 제거된 배경
        "inference_ms": round(infer_ms, 1)
    })


# ============================================================
# 4. 서버 시작
# ============================================================
if __name__ == '__main__':
    print("=" * 65)
    print("🎯 SAM3 정밀 분할 서버 — RTX 4090")
    print("=" * 65)

    # SAM3 모델 로딩
    sam3_engine, SAM3_BACKEND = load_sam3_model()

    print(f"\n🌐 SAM3 서버 시작: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"   NUC app.py의 SAM3_SERVER_URL을 http://<이 PC IP>:{FLASK_PORT}/segment 로 설정")
    print(f"📡 엔드포인트: POST /segment (박스→마스크), GET /ping (헬스체크)")
    print("=" * 65)

    # Flask 실행 (threaded=True로 동시 요청 처리)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
