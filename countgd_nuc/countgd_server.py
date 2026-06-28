# CountGD 추론 Flask 서버 — 4090에서 텍스트 프롬프트로 객체를 카운팅 (포트 5005)
# [2026-06-28 작성] 컵 재고조사 파이프라인의 카운팅 단계 담당
# CountGD 공식 레포(niki-amini-naieni/CountGD)의 추론 로직을 REST API로 래핑

import io
import base64
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify

# CountGD 레포 내부 모듈 (이 서버는 CountGD 레포 루트에서 실행해야 함)
from models import build_model                       # CountGD 모델 빌더
from util.slconfig import SLConfig                    # 설정 로더
import datasets.transforms as T                       # 입력 전처리 transform

# ============================================================
# ★ 사용자 환경에 맞게 수정할 설정값들
# ============================================================
CONFIG_PATH = "config/cfg_fsc147_test.py"            # ← CountGD 설정 파일
CHECKPOINT_PATH = "checkpoints/checkpoint_fsc147_best.pth"  # ← 학습된 가중치
BERT_PATH = "checkpoints/bert-base-uncased"          # ← 텍스트 인코더 경로
DEVICE = "cuda"                                       # ← 4090 GPU 사용
CONF_THRESH = 0.23                                    # ← 카운팅 신뢰도 임계값 (논문 기본값)
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5005                                     # ← CountGD 전용 포트 (빅맨님 5001~5004와 구분)

app = Flask(__name__)
model = None          # CountGD 모델 (전역)
transform = None      # 입력 전처리 (전역)


# ============================================================
# 1. 모델 로딩
# ============================================================
def load_countgd():
    """CountGD 모델과 전처리 transform을 로드한다."""
    global model, transform

    # 설정 로드 + 텍스트 인코더 경로 주입
    cfg = SLConfig.fromfile(CONFIG_PATH)              # 설정 파일 읽기
    cfg.text_encoder_type = BERT_PATH                 # BERT 경로 지정 (재다운로드 방지)
    cfg.device = DEVICE                               # GPU 지정

    # 모델 빌드 + 가중치 로드
    model_built, _, _ = build_model(cfg)              # CountGD 모델 생성
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")  # 가중치 로드
    model_built.load_state_dict(checkpoint["model"], strict=False)  # 가중치 주입
    model_built.eval().to(DEVICE)                     # 평가 모드 + GPU 이동
    model = model_built

    # 입력 전처리: 짧은 변 800px 리사이즈 + 정규화 (논문 설정)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),         # 짧은 변 800 리사이즈
        T.ToTensor(),                                 # 텐서 변환
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 정규화
    ])
    print(f"✅ CountGD 로드 완료 (device={DEVICE})")


# ============================================================
# 2. 텍스트 프롬프트 카운팅
# ============================================================
@torch.inference_mode()                               # 추론 전용 (메모리 절약)
def count_objects(image_pil, text_prompt, conf=CONF_THRESH):
    """이미지에서 텍스트로 지정한 객체를 카운팅한다.

    Args:
        image_pil: PIL 이미지
        text_prompt: 카운팅 대상 텍스트 (예: "cup")
        conf: 신뢰도 임계값
    Returns:
        (개수, 박스리스트, 점수리스트)
    """
    # 프롬프트 끝에 마침표 추가 (GroundingDINO 규약)
    if not text_prompt.endswith("."):
        text_prompt = text_prompt + " ."

    # 전처리
    input_tensor, _ = transform(image_pil, None)      # 리사이즈+정규화
    input_tensor = input_tensor.to(DEVICE)            # GPU 이동

    # 추론 (autocast로 bfloat16 — 4090 가속)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_tensor[None],                       # 배치 차원 추가
            captions=[text_prompt],                   # 텍스트 프롬프트
            exemplars=[torch.tensor([]).to(DEVICE)],  # 예시 박스 없음 (텍스트 전용)
            labels=[torch.tensor([]).to(DEVICE)],
        )

    # 후처리: 신뢰도 임계값 넘는 박스만 카운팅
    logits = outputs["pred_logits"].sigmoid()[0]      # 박스별 신뢰도
    boxes = outputs["pred_boxes"][0]                  # 박스 좌표 (cx,cy,w,h 정규화)
    scores = logits.max(dim=-1).values                # 박스별 최대 점수
    keep = scores > conf                              # 임계값 필터
    kept_boxes = boxes[keep].cpu().numpy()            # 남은 박스
    kept_scores = scores[keep].cpu().numpy()          # 남은 점수

    count = int(keep.sum())                           # 박스 개수 = 카운트
    return count, kept_boxes.tolist(), kept_scores.tolist()


# ============================================================
# 3. Flask 라우트
# ============================================================
def _decode_image(b64):
    """base64 → PIL 이미지 디코딩."""
    raw = base64.b64decode(b64)                       # base64 디코딩
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)  # → BGR
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)        # BGR → RGB
    return Image.fromarray(rgb)                        # PIL 변환


@app.route("/count", methods=["POST"])
def count_route():
    """텍스트 프롬프트로 객체를 카운팅한다.

    Request JSON: {"image": base64, "text": "cup", "conf": 0.23}
    Response JSON: {"count": N, "boxes": [...], "scores": [...]}
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "image(base64) 필요"}), 400

    text = data.get("text", "cup")                    # 카운팅 대상
    conf = float(data.get("conf", CONF_THRESH))       # 신뢰도 임계값

    image_pil = _decode_image(data["image"])          # 이미지 디코딩
    count, boxes, scores = count_objects(image_pil, text, conf)  # 카운팅 실행

    print(f"🔢 CountGD: \"{text}\" → {count}개 (conf>{conf})")
    return jsonify({
        "count": count,                               # 총 개수
        "boxes": boxes,                               # 정규화 박스 (cx,cy,w,h)
        "scores": scores,                             # 신뢰도
        "text": text
    })


@app.route("/health")
def health():
    """서버 상태 확인."""
    return jsonify({"status": "ok", "model_loaded": model is not None, "device": DEVICE})


# ============================================================
# 4. 서버 시작
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔢 CountGD 카운팅 서버 — 텍스트 프롬프트 객체 카운팅")
    print("   [v2026-06-28 재고조사판] 포트 5005")
    print("=" * 60)

    load_countgd()                                    # 모델 로드

    print(f"🌐 서버 시작: http://{FLASK_HOST}:{FLASK_PORT}")
    print("=" * 60)
    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)
