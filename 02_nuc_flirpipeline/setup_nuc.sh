#!/bin/bash
# ============================================================
# 파일명: setup_nuc.sh
# 설명: NUC 메인 파이프라인 환경 설치 (conda lerobot 환경 재활용)
# 사용법: chmod +x setup_nuc.sh && ./setup_nuc.sh
# 변경 이력:
#   - [2026-06-22] 최초 생성
# ============================================================
set -e

echo "======================================"
echo " 🖥️ NUC 메인 파이프라인 설치"
echo "======================================"

# --- [1단계] conda 환경 활성화 (기존 lerobot 재활용) ---
echo ""
echo "[1/4] conda lerobot 환경 활성화..."
# conda init이 안 되어 있으면 먼저 실행 (Stage1 노하우)
source ~/miniconda3/bin/activate lerobot || {
    echo "⚠️ conda init 필요 → 실행합니다"
    ~/miniconda3/bin/conda init bash && source ~/.bashrc
    conda activate lerobot
}

# --- [2단계] 패키지 설치 ---
echo ""
echo "[2/4] Python 패키지 설치..."
pip install -r requirements.txt --break-system-packages

# --- [3단계] 열화상 카메라 권한 확인 ---
echo ""
echo "[3/4] 열화상 카메라(/dev/video0) 확인..."
if [ -e /dev/video0 ]; then
    echo "  ✅ /dev/video0 존재"
    # 카메라 점유 프로세스 확인 (Stage1 노하우)
    if sudo fuser /dev/video0 2>/dev/null; then
        echo "  ⚠️ 카메라가 다른 프로세스에 점유됨 → 종료 필요"
    fi
else
    echo "  ❌ /dev/video0 없음 → FLIR/Seek USB 연결 확인"
fi

# --- [4단계] 완료 ---
echo ""
echo "[4/4] 설치 완료!"
echo "======================================"
echo "📌 실행 전 config/config.yaml 확인:"
echo "   - yolo.model_path : 학습된 OpenVINO 모델 절대경로"
echo "   - sam3.server_url : 4090 IP (192.168.0.75:5001)"
echo "   - irgpt.server_url: 4070 IP (192.168.0.74:5002)"
echo "   - thermal.raw_scale/offset: 센서 보정값"
echo ""
echo "📌 실행:"
echo "   python app.py"
echo "📌 브라우저: http://$(hostname -I | awk '{print $1}'):5000"
