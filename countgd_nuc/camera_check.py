# 누크 웹카메라 진단 도구 — 어느 장치 번호로 카메라가 열리는지 자동으로 찾아준다
# [2026-06-28 작성] 재고조사 시스템 시작 전, 카메라가 정상인지 확인하는 1차 점검 스크립트

import argparse   # 명령줄 옵션(--index, --test, --resolution)을 받기 위함
import time
import sys

import cv2        # OpenCV로 카메라를 열고 프레임을 읽는다


def scan_cameras(max_index=5):
    """0번부터 max_index번까지 열어보며 살아있는 카메라 번호를 찾는다."""
    print("=" * 52)
    print("🔍 카메라 장치 검색 (0 ~ %d번)" % max_index)
    print("=" * 52)
    found = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)                  # idx번 카메라 열기 시도
        if cap.isOpened():
            ok, frame = cap.read()                   # 실제 프레임이 읽히는지 확인
            if ok and frame is not None:
                h, w = frame.shape[:2]               # 해상도 확인
                print(f"  ✅ /dev/video{idx}  열림 — 해상도 {w}x{h}")
                found.append(idx)
            else:
                print(f"  ⚠️ /dev/video{idx}  열렸지만 프레임 없음")
            cap.release()                            # 반드시 해제 (점유 방지)
        else:
            print(f"  ❌ /dev/video{idx}  없음")
    print("-" * 52)
    if found:
        print(f"👉 사용 가능한 카메라 번호: {found}")
        print(f"   app_step.py의 CAM_INDEX를 {found[0]}로 설정하세요.")
    else:
        print("⚠️ 카메라를 찾지 못했습니다. USB 연결과 권한을 확인하세요.")
    print("=" * 52)
    return found


def test_capture(index, seconds=5):
    """지정한 카메라에서 몇 초간 프레임을 읽어 FPS와 안정성을 측정한다."""
    print(f"📷 카메라 {index}번 테스트 ({seconds}초간 프레임 읽기)")
    cap = cv2.VideoCapture(index)                    # 카메라 열기
    if not cap.isOpened():
        print(f"❌ 카메라 {index}번을 열 수 없습니다.")
        return
    frames = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()                       # 프레임 읽기
        if ok:
            frames += 1                              # 성공한 프레임 수 세기
    cap.release()
    elapsed = time.time() - t0
    fps = frames / elapsed if elapsed > 0 else 0
    print(f"  ▶ {frames}프레임 / {elapsed:.1f}초 = 평균 {fps:.1f} FPS")
    if fps >= 10:
        print("  ✅ 정상 — 실시간 사용에 충분합니다.")
    elif fps > 0:
        print("  ⚠️ 느림 — USB 대역폭이나 해상도를 확인하세요.")
    else:
        print("  ❌ 프레임을 못 읽었습니다.")


def test_resolution(index, width, height):
    """원하는 해상도를 카메라가 실제로 지원하는지 확인한다."""
    print(f"📐 카메라 {index}번에 {width}x{height} 해상도 요청")
    cap = cv2.VideoCapture(index)                    # 카메라 열기
    if not cap.isOpened():
        print(f"❌ 카메라 {index}번을 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)         # 가로 해상도 요청
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)       # 세로 해상도 요청
    ok, frame = cap.read()                           # 한 프레임 읽어 실제 크기 확인
    cap.release()
    if ok and frame is not None:
        rh, rw = frame.shape[:2]                      # 실제로 적용된 해상도
        if (rw, rh) == (width, height):
            print(f"  ✅ {rw}x{rh} 그대로 지원됩니다.")
        else:
            print(f"  ⚠️ 요청 {width}x{height} → 실제 {rw}x{rh}로 조정됨")
            print(f"     app_step.py의 CAM_WIDTH/HEIGHT를 {rw}/{rh}로 맞추세요.")
    else:
        print("  ❌ 프레임을 읽지 못했습니다.")


def main():
    """명령줄 옵션에 따라 검색/테스트/해상도 확인을 실행한다."""
    parser = argparse.ArgumentParser(description="누크 웹카메라 진단 도구")
    parser.add_argument("--scan", action="store_true", help="사용 가능한 카메라 번호 모두 검색")
    parser.add_argument("--test", type=int, metavar="N", help="N번 카메라 5초 FPS 테스트")
    parser.add_argument("--resolution", type=int, metavar="N", help="N번 카메라 해상도 지원 확인")
    parser.add_argument("--width", type=int, default=1280, help="확인할 가로 해상도 (기본 1280)")
    parser.add_argument("--height", type=int, default=720, help="확인할 세로 해상도 (기본 720)")
    args = parser.parse_args()

    # 옵션이 하나도 없으면 기본으로 전체 검색 실행
    if not (args.scan or args.test is not None or args.resolution is not None):
        scan_cameras()
        return

    if args.scan:
        scan_cameras()
    if args.test is not None:
        test_capture(args.test)
    if args.resolution is not None:
        test_resolution(args.resolution, args.width, args.height)


if __name__ == "__main__":
    main()   # 스크립트로 직접 실행될 때 main() 호출
