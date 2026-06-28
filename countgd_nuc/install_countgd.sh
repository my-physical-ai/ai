#!/bin/bash
# CountGD 설치 스크립트 — 4090 PC (sam3 환경 별도 권장)
# [2026-06-28] 컵 재고조사 카운팅 서버 구축
set -e  # 오류 발생 시 즉시 중단

echo "=========================================="
echo " CountGD 설치 시작 (4090 / Ubuntu)"
echo "=========================================="

# ── 1. 컴파일러 설치 (GroundingDINO 커스텀 ops 빌드에 필수) ──
echo "[1/7] GCC 11 설치..."
sudo apt update                                    # 패키지 목록 갱신
sudo apt install -y build-essential gcc-11 g++-11  # GCC 11 설치 (CountGD 권장 버전)

# ── 2. conda 환경 생성 (sam3와 충돌 방지 위해 전용 환경 권장) ──
echo "[2/7] conda 환경 생성 (countgd)..."
conda create -n countgd python=3.10 -y             # Python 3.10 전용 환경
source activate countgd || conda activate countgd  # 환경 활성화

# ── 3. CountGD 레포 클론 ──
echo "[3/7] CountGD 레포 클론..."
git clone https://github.com/niki-amini-naieni/CountGD.git
cd CountGD

# ── 4. PyTorch + 의존성 설치 (4090은 CUDA 12.x) ──
echo "[4/7] PyTorch 및 의존성 설치..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install -r requirements.txt                    # CountGD 의존성
pip install flask gdown                             # 서버 + 다운로드 도구

# ── 5. GroundingDINO 커스텀 ops 컴파일 (★ 가장 중요) ──
echo "[5/7] GroundingDINO ops 컴파일 (gcc-11 강제)..."
export CC=/usr/bin/gcc-11                           # gcc 11 사용 강제
export CXX=/usr/bin/g++-11
cd models/GroundingDINO/ops
python setup.py build install                       # 커스텀 CUDA 연산 빌드
python test.py                                      # 검증: 6줄 모두 "True" 나와야 정상
cd ../../../

# ── 6. 체크포인트 다운로드 ──
echo "[6/7] 모델 가중치 다운로드..."
mkdir -p checkpoints
# CountGD FSC-147 학습 가중치 (gdown으로 구글드라이브에서)
gdown --id 1Zob_z5ghBoUUTjsdgWdg2XR0d9-tShze -O checkpoints/checkpoint_fsc147_best.pth || \
  echo "⚠️ 가중치 자동 다운로드 실패 — README의 수동 링크 사용하세요"

# BERT 텍스트 인코더 (재다운로드 방지용 로컬 저장)
python -c "
from transformers import BertModel, BertTokenizer  # BERT 다운로드
BertModel.from_pretrained('bert-base-uncased').save_pretrained('checkpoints/bert-base-uncased')
BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained('checkpoints/bert-base-uncased')
print('✅ BERT 저장 완료')
"

# ── 7. 서버 파일 배치 ──
echo "[7/7] 서버 파일 복사..."
echo "  countgd_server.py 를 CountGD 레포 루트(현재 위치)에 복사하세요"

echo "=========================================="
echo " ✅ 설치 완료!"
echo " 실행: python countgd_server.py"
echo "=========================================="
