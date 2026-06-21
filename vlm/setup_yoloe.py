# YOLOE-26 모델 다운로드 + OpenVINO 변환 — NUC에서 수업 전 실행
# [2026-06-21 작성] YOLO26(80종 고정) → YOLOE-26(텍스트로 무엇이든 탐지)
#
# 실행: conda activate lerobot && python setup_yoloe.py

import time

# ============================================================
# ★ 품질검사 대상 물체 클래스 정의
# ============================================================
DETECT_CLASSES = [
    # 물류/품질검사 대상
    "cup", "bottle", "box", "package", "can", "container", "bag",
    "carton", "crate", "parcel", "envelope",

    # 음료/용기
    "mug", "glass", "water bottle", "coffee cup", "tumbler", "jar",

    # 일반 사무실 물체
    "person", "chair", "table", "desk", "monitor", "keyboard", "mouse",
    "phone", "laptop", "book", "pen", "paper", "folder",

    # 산업/안전
    "helmet", "vest", "glove", "tool", "wire", "pipe",
]

# 모델 크기 선택 (NUC CPU 기준)
# [2026-06-21 수정] YOLOE 사전학습 모델은 모두 -seg.pt 형식!
# yoloe-26s-seg.pt: 균형잡힌 성능 (추천, 탐지+세그멘테이션 모두 지원)
# yoloe-26n-seg.pt: 더 빠르지만 정확도 낮음 (초경량)
MODEL_NAME = "yoloe-26s-seg.pt"


def main():
    print("=" * 60)
    print("🎯 YOLOE-26 Open Vocabulary 모델 설정")
    print(f"   모델: {MODEL_NAME}")
    print(f"   탐지 클래스: {len(DETECT_CLASSES)}종")
    print("=" * 60)

    # [1] YOLOE 모델 다운로드 + 클래스 설정
    print(f"\n[1/3] YOLOE 모델 다운로드 중...")
    t0 = time.time()

    from ultralytics import YOLOE

    model = YOLOE(MODEL_NAME)                          # 자동 다운로드
    model.set_classes(DETECT_CLASSES)                   # 탐지 대상 설정!
    print(f"✅ 모델 로드 완료 ({time.time()-t0:.1f}초)")
    print(f"   설정된 클래스: {DETECT_CLASSES[:5]}... 외 {len(DETECT_CLASSES)-5}종")

    # [2] OpenVINO 변환 (NUC CPU 가속용)
    print(f"\n[2/3] OpenVINO 변환 중 (CPU 가속)...")
    t0 = time.time()

    export_path = model.export(format="openvino", imgsz=640)
    print(f"✅ OpenVINO 변환 완료 ({time.time()-t0:.1f}초)")
    print(f"   저장 위치: {export_path}")

    # [3] 변환된 모델로 테스트 추론
    print(f"\n[3/3] 테스트 추론...")
    import numpy as np

    # 더미 이미지로 테스트 (640x480 검정)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    from ultralytics import YOLO
    ov_model = YOLO(export_path)

    t0 = time.time()
    results = ov_model(dummy, verbose=False, imgsz=640)
    first_ms = (time.time()-t0) * 1000

    t0 = time.time()
    results = ov_model(dummy, verbose=False, imgsz=640)
    second_ms = (time.time()-t0) * 1000

    print(f"✅ 추론 성공!")
    print(f"   첫 추론: {first_ms:.0f}ms (워밍업)")
    print(f"   두번째:  {second_ms:.0f}ms (실제 속도)")

    # 클래스 목록 확인
    print(f"\n📋 탐지 가능 클래스 ({len(ov_model.names)}종):")
    for i, name in ov_model.names.items():
        print(f"   {i}: {name}")

    print("\n" + "=" * 60)
    print("✅ 설정 완료!")
    print(f"   app_scenario3.py의 YOLO_MODEL_PATH를 아래로 변경하세요:")
    print(f"   YOLO_MODEL_PATH = \"{export_path}\"")
    print("=" * 60)


if __name__ == '__main__':
    main()
