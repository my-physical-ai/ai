# multi_send_camera.py — Pi에서 2대 USB 카메라를 ZeroMQ로 NUC에 전송
# [2026-02-16] 카메라별 독립 스레드 + 독립 포트 + 정확한 타이밍 제어
"""
LeKiwi Multi-Camera Sender (Pi5) — ZeroMQ 버전
================================================
Pi에 장착된 2대 USB 카메라의 JPEG 프레임을 각각 다른 포트로 전송합니다.

설계 개념
--------
  [Pi5]  FRONT 카메라 → ZeroMQ PUB :5556  ─→  [NUC] SUB 수신 (YOLO/ACT)
         TOP   카메라 → ZeroMQ PUB :5557  ─→  [NUC] SUB 수신 (장애물 예측)

  ※ 포트를 분리하는 이유: ZeroMQ CONFLATE는 단일 프레임 메시지만 지원.
    멀티파트(토픽+프레임)를 쓰면 CONFLATE와 충돌하여 크래시 발생!
    → 카메라별 독립 포트가 가장 안전한 방식.

카메라 역할
  FRONT (/dev/video0) → 정면/바닥 시점 (YOLO 탐지, ACT 데이터 수집)
  TOP   (/dev/video2) → 상단/전방 시점 (장애물 예측, 환경 인식)

camera_server_pi5_v2.py 와의 차이
  - TCP 소켓 대신 ZeroMQ PUB-SUB 사용
  - NUC에서 CONFLATE=1로 최신 프레임만 유지 (딜레이 없음)
  - TCP는 모든 프레임을 순서대로 전달 (데이터 수집에 적합)
  - ZeroMQ는 프레임 버려도 OK인 실시간 탐지에 최적

실행
  python3 multi_send_camera.py
  python3 multi_send_camera.py --front /dev/video1 --top /dev/video2
  python3 multi_send_camera.py --fps 25
"""

import argparse
import os
import signal
import sys
import threading
import time
import logging

import cv2
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 기본 설정 ────────────────────────────────────────────
FRONT_DEVICE = "/dev/video0"    # FRONT 카메라 장치
TOP_DEVICE   = "/dev/video2"    # TOP 카메라 장치
FRONT_PORT   = 5556             # FRONT ZeroMQ 포트
TOP_PORT     = 5557             # TOP ZeroMQ 포트
WIDTH        = 640              # 해상도 가로
HEIGHT       = 480              # 해상도 세로
TARGET_FPS   = 15               # 목표 FPS
JPEG_QUALITY = 80               # JPEG 압축 품질 (1~100)


def camera_sender(name: str, device: str, port: int,
                  fps: int, stop_event: threading.Event) -> None:
    """단일 카메라 전송 스레드 — ZeroMQ PUB으로 JPEG 프레임 발행.

    정확한 타이밍 제어(next_time 방식)로 목표 FPS를 정확히 유지한다.
    카메라 열기 실패 시 3초마다 재시도한다.

    Args:
        name: 카메라 이름 (로그용, "FRONT"/"TOP")
        device: 장치 경로 (/dev/video0 등)
        port: ZeroMQ PUB 바인딩 포트
        fps: 목표 FPS
        stop_event: 종료 신호
    """
    # ── ZeroMQ PUB 소켓 생성 ──
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)          # PUB = 발행자
    sock.bind(f"tcp://*:{port}")        # 모든 IP에서 수신 허용
    log.info(f"[{name}] ZeroMQ PUB 시작: tcp://*:{port} (장치 {device})")

    interval = 1.0 / fps               # 프레임 간격 (초)

    while not stop_event.is_set():
        # ── 카메라 열기 (실패 시 재시도) ──
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 최신 프레임 우선

        if not cap.isOpened():
            log.error(f"[{name}] 카메라 열기 실패: {device} — 3초 후 재시도")
            cap.release()
            stop_event.wait(3.0)
            continue

        log.info(f"[{name}] 카메라 열림! {WIDTH}x{HEIGHT} @ {fps}fps")

        count = 0
        start = time.time()
        next_time = start               # 다음 프레임 전송 시각

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()  # 카메라에서 1프레임 읽기
                if not ret:
                    log.warning(f"[{name}] 프레임 읽기 실패 — 카메라 재열기")
                    break                # 카메라 재열기 루프로 이동

                # JPEG 압축 후 ZeroMQ로 전송
                _, buf = cv2.imencode('.jpg', frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                sock.send(buf.tobytes())  # 단일 프레임 전송 (CONFLATE 호환!)
                count += 1

                # 30프레임마다 실제 FPS 출력
                if count % 30 == 0:
                    actual_fps = count / (time.time() - start)
                    log.info(f"[{name}] 전송 {count}프레임 | 실제 {actual_fps:.1f}fps")

                # 정확한 타이밍 제어 (작업시간 자동 보상)
                next_time += interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cap.release()

    sock.close()
    ctx.term()
    log.info(f"[{name}] 전송 종료")


def check_device(device: str) -> bool:
    """카메라 장치 존재 여부를 확인한다."""
    return os.path.exists(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeKiwi Multi-Camera Sender (Pi5, ZeroMQ)")
    parser.add_argument("--front", default=FRONT_DEVICE,
                        help=f"FRONT 카메라 장치 (기본 {FRONT_DEVICE})")
    parser.add_argument("--top", default=TOP_DEVICE,
                        help=f"TOP 카메라 장치 (기본 {TOP_DEVICE})")
    parser.add_argument("--front-port", type=int, default=FRONT_PORT,
                        help=f"FRONT ZeroMQ 포트 (기본 {FRONT_PORT})")
    parser.add_argument("--top-port", type=int, default=TOP_PORT,
                        help=f"TOP ZeroMQ 포트 (기본 {TOP_PORT})")
    parser.add_argument("--fps", type=int, default=TARGET_FPS,
                        help=f"목표 FPS (기본 {TARGET_FPS})")
    args = parser.parse_args()

    # ── 카메라 장치 확인 ──
    for dev, name in [(args.front, "FRONT"), (args.top, "TOP")]:
        if not check_device(dev):
            log.error(f"❌ {name} 카메라 장치 없음: {dev}")
            log.error("   ls /dev/video*  또는  v4l2-ctl --list-devices 로 확인")
            sys.exit(1)

    # ── 종료 처리 ──
    stop_event = threading.Event()

    def handle_signal(sig, frame):
        log.info("종료 신호 수신...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── 카메라별 전송 스레드 시작 ──
    threads = [
        threading.Thread(target=camera_sender,
                         args=("FRONT", args.front, args.front_port,
                               args.fps, stop_event),
                         daemon=True),
        threading.Thread(target=camera_sender,
                         args=("TOP", args.top, args.top_port,
                               args.fps, stop_event),
                         daemon=True),
    ]
    for t in threads:
        t.start()

    # ── 시작 안내 ──
    print(f"\n{'='*60}")
    print(f"  📹 LeKiwi Multi-Camera Sender (ZeroMQ)")
    print(f"  FRONT : {args.front:14s} → ZeroMQ PUB :{args.front_port}")
    print(f"  TOP   : {args.top:14s} → ZeroMQ PUB :{args.top_port}")
    print(f"  FPS   : {args.fps}")
    print(f"")
    print(f"  NUC 수신 예시:")
    print(f"    FRONT: sock.connect('tcp://PI_IP:{args.front_port}')")
    print(f"    TOP  : sock.connect('tcp://PI_IP:{args.top_port}')")
    print(f"{'='*60}")
    print(f"  종료: Ctrl+C\n")

    # 메인 스레드는 종료 신호 대기
    stop_event.wait()
    log.info("모든 카메라 전송 종료 완료")


if __name__ == "__main__":
    main()
