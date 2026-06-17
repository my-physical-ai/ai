# basic_detect.py — Pi 영상을 수신하여 기본 PyTorch YOLO26으로 탐지 (1대 또는 2대)
# [2026-02-16] 싱글: FRONT만 / 멀티: FRONT+TOP 나란히 표시 + 각각 YOLO 탐지
"""
실행 예시:
  python3 basic_detect.py                          # 싱글 (FRONT만)
  python3 basic_detect.py --multi                  # 멀티 (FRONT+TOP 나란히)
  python3 basic_detect.py --pi-ip 192.168.50.111   # IP 변경
"""

import argparse
import threading
import time

import cv2
import zmq
import numpy as np
from ultralytics import YOLO


def single_detect(pi_ip: str, port: int, model_path: str):
    """싱글카메라 YOLO 탐지 — 1개 창에 표시."""
    model = YOLO(model_path)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{pi_ip}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f"\n🎯 싱글카메라 PyTorch YOLO 탐지 시작!")
    print(f"📡 Pi: tcp://{pi_ip}:{port}")
    print(f"{'프레임':>6} | {'수신':>8} | {'YOLO':>8} | {'합계':>8} | {'FPS':>6}")
    print("-" * 55)

    count = 0
    fps_start = time.time()

    while True:
        t0 = time.time()
        buf = sock.recv()
        t1 = time.time()

        frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        t2 = time.time()

        yolo_ms = (t2 - t1) * 1000
        cv2.putText(annotated, f"PyTorch: {yolo_ms:.0f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("YOLO26 (PyTorch)", annotated)
        count += 1

        if count % 15 == 0:
            fps = count / (time.time() - fps_start)
            print(f"{count:>6} | {(t1-t0)*1000:>6.1f}ms | {yolo_ms:>6.1f}ms | {(t2-t0)*1000:>6.1f}ms | {fps:>5.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n📊 결과: 평균 {count/(time.time()-fps_start):.1f} FPS")
    print(f"💡 OpenVINO를 적용하면 2~5배 빨라져요!")


def multi_detect(pi_ip: str, front_port: int, top_port: int, model_path: str):
    """멀티카메라 YOLO 탐지 — FRONT+TOP 나란히 표시."""
    model = YOLO(model_path)

    # ── 2개 ZeroMQ 소켓 ──
    ctx = zmq.Context()
    front_sock = ctx.socket(zmq.SUB)
    front_sock.connect(f"tcp://{pi_ip}:{front_port}")
    front_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    front_sock.setsockopt(zmq.CONFLATE, 1)       # 최신 프레임만
    front_sock.setsockopt(zmq.RCVTIMEO, 1000)    # 1초 타임아웃

    top_sock = ctx.socket(zmq.SUB)
    top_sock.connect(f"tcp://{pi_ip}:{top_port}")
    top_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    top_sock.setsockopt(zmq.CONFLATE, 1)         # 최신 프레임만
    top_sock.setsockopt(zmq.RCVTIMEO, 1000)      # 1초 타임아웃

    print(f"\n🎯 멀티카메라 PyTorch YOLO 탐지 시작!")
    print(f"📡 FRONT: tcp://{pi_ip}:{front_port}")
    print(f"📡 TOP  : tcp://{pi_ip}:{top_port}")
    print(f"{'프레임':>6} | {'FRONT':>8} | {'TOP':>8} | {'FPS':>6}")
    print("-" * 45)

    # 대기 이미지 (카메라 미수신 시)
    waiting = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(waiting, "Waiting...", (200, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)

    count = 0
    fps_start = time.time()

    while True:
        # ── FRONT 프레임 수신 + 탐지 ──
        try:
            buf = front_sock.recv()
            frame_f = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            t0 = time.time()
            res_f = model(frame_f, verbose=False)
            front_ms = (time.time() - t0) * 1000
            ann_f = res_f[0].plot()
            cv2.putText(ann_f, f"FRONT {front_ms:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except zmq.Again:
            ann_f = waiting.copy()
            cv2.putText(ann_f, "FRONT: No signal", (150, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            front_ms = 0

        # ── TOP 프레임 수신 + 탐지 ──
        try:
            buf = top_sock.recv()
            frame_t = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            t0 = time.time()
            res_t = model(frame_t, verbose=False)
            top_ms = (time.time() - t0) * 1000
            ann_t = res_t[0].plot()
            cv2.putText(ann_t, f"TOP {top_ms:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        except zmq.Again:
            ann_t = waiting.copy()
            cv2.putText(ann_t, "TOP: No signal", (170, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            top_ms = 0

        # ── 나란히 합쳐서 표시 ──
        combined = np.hstack([ann_f, ann_t])      # 1280x480
        cv2.imshow("YOLO26 Multi-Camera (PyTorch)", combined)
        count += 1

        if count % 15 == 0:
            fps = count / (time.time() - fps_start)
            print(f"{count:>6} | {front_ms:>6.1f}ms | {top_ms:>6.1f}ms | {fps:>5.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n📊 결과: 평균 {count/(time.time()-fps_start):.1f} FPS")


def main():
    parser = argparse.ArgumentParser(description="YOLO 기본 탐지 (싱글/멀티)")
    parser.add_argument("--pi-ip", default="192.168.50.111",
                        help="Pi IP 주소")
    parser.add_argument("--multi", action="store_true",
                        help="멀티카메라 모드 (FRONT+TOP)")
    parser.add_argument("--front-port", type=int, default=5556)
    parser.add_argument("--top-port", type=int, default=5557)
    parser.add_argument("--model", default="yolo26n.pt",
                        help="YOLO 모델 (기본 yolo26n.pt)")
    args = parser.parse_args()

    if args.multi:
        multi_detect(args.pi_ip, args.front_port, args.top_port, args.model)
    else:
        single_detect(args.pi_ip, args.front_port, args.model)


if __name__ == "__main__":
    main()
