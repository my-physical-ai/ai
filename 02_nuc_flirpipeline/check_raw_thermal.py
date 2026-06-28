# ============================================================
# 파일명: check_raw_thermal.py
# 설명: /dev/video3가 raw 16비트(Y16)인지 진단하고, raw면 절대온도(℃)로 변환
# 목적: 컬러맵 RGB 대신 raw 데이터를 받을 수 있는지 확인 (절대온도의 핵심)
# 사용: python check_raw_thermal.py
# ============================================================

import subprocess                                       # v4l2 포맷 확인용
import cv2                                              # 영상 캡처
import numpy as np                                      # 배열 연산

DEVICE_INDEX = 3                                         # FLIR ONE 열화상 장치 번호


def list_v4l2_formats(dev_index):
    """v4l2-ctl로 해당 장치가 지원하는 픽셀 포맷을 출력한다 (Y16=raw 가능 신호)."""
    dev = f"/dev/video{dev_index}"                       # 장치 경로
    try:
        out = subprocess.check_output(                   # 지원 포맷 질의
            ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
            stderr=subprocess.STDOUT, text=True)
        print(out)                                       # 포맷 목록 출력
        # Y16 / GREY 가 있으면 raw 16비트 추출 가능
        if "Y16" in out:                                 # 16비트 그레이 지원 시
            print("✅ Y16 지원 — raw 16비트 절대온도 추출 가능!")
        elif "GREY" in out:                              # 8비트 그레이만
            print("⚠️ GREY(8비트)만 지원 — 정밀도 낮음, RGB보단 나음")
        else:                                            # RGB/BGR만
            print("❌ raw 미지원 — 드라이버가 컬러맵 RGB만 출력 (역변환 필요)")
    except FileNotFoundError:                            # v4l2-ctl 미설치
        print("v4l2-ctl 없음 → 설치: sudo apt install v4l-utils")
    except subprocess.CalledProcessError as e:           # 질의 실패
        print(f"포맷 질의 실패: {e.output}")


def try_capture_y16(dev_index):
    """장치를 Y16(raw 16비트)으로 강제 캡처하여 절대온도 변환을 시도한다."""
    cap = cv2.VideoCapture(dev_index)                    # 장치 열기
    # raw 16비트 강제 설정 (드라이버가 지원할 때만 적용됨)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)                 # RGB 자동변환 끄기 (핵심!)
    cap.set(cv2.CAP_PROP_FOURCC,                         # Y16 포맷 강제
            cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
    ret, frame = cap.read()                              # 한 프레임 읽기
    if not ret or frame is None:                         # 실패 시
        print("❌ 캡처 실패 — flirone 드라이버 실행 확인")
        cap.release()
        return None
    print(f"캡처 프레임 shape={frame.shape}, dtype={frame.dtype}")  # 형태 출력
    # raw 16비트면 dtype=uint16 또는 단일채널 — 절대온도 변환 가능
    if frame.dtype == np.uint16 or (frame.ndim == 2):    # raw 데이터 판정
        raw = frame.astype(np.float32)                   # 실수 변환
        # FLIR Lepton 코어 선형 근사: 켈빈 = raw * 0.01, 섭씨 = 켈빈 - 273.15
        celsius = raw * 0.01 - 273.15                    # raw → 섭씨 근사
        print(f"✅ raw 16비트! 온도범위 {celsius.min():.1f}~{celsius.max():.1f}℃")
        cap.release()
        return celsius                                   # 픽셀별 섭씨 배열 반환
    else:                                                # RGB로 나오면
        print("❌ RGB로 디코딩됨 — 이 장치는 raw 미지원, 해결책 A 사용")
        cap.release()
        return None


if __name__ == "__main__":
    print("=" * 55)
    print("FLIR ONE Gen2 raw 16비트 진단")
    print("=" * 55)
    list_v4l2_formats(DEVICE_INDEX)                      # 1단계: 포맷 확인
    print("-" * 55)
    try_capture_y16(DEVICE_INDEX)                        # 2단계: raw 캡처 시도
