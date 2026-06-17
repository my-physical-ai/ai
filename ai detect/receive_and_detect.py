# receive_and_detect.py — Pi 영상 수신 + OpenVINO YOLO26 탐지 (싱글/멀티)
# [2026-02-16] 싱글: FRONT만 / 멀티: FRONT+TOP 나란히 + 각각 YOLO 탐지
"""
실행 예시:
  python3 receive_and_detect.py                    # 싱글 (FRONT만)
  python3 receive_and_detect.py --multi            # 멀티 (FRONT+TOP 나란히)
  python3 receive_and_detect.py --pi-ip 192.168.50.111
"""

import argparse
import time

import cv2
import zmq
import numpy as np
from ultralytics import YOLO

# ── 기본 설정 ──
DEFAULT_PI_IP = "192.168.50.111"
DEFAULT_MODEL = "yolo26n_openvino_model/"
DEFAULT_IMGSZ = 640


def single_detect(pi_ip: str, port: int, model_path: str, imgsz: int):
    """싱글카메라 OpenVINO YOLO 탐지."""
    print(f"⚡ YOLO 모델 로딩: {model_path}")
    model = YOLO(model_path, task="detect")
    print("✅ 모델 준비 완료!")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{pi_ip}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, 1)          # 최신 프레임만 (딜레이 방지)

    print(f"\n🚀 OpenVINO 싱글카메라 탐지 시작!")
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
        results = model(frame, verbose=False, imgsz=imgsz)
        annotated = results[0].plot()
        t2 = time.time()

        cv2.imshow("YOLO26 OpenVINO", annotated)
        count += 1

        if count % 15 == 0:
            fps = count / (time.time() - fps_start)
            print(f"{count:>6} | {(t1-t0)*1000:>6.1f}ms | {(t2-t1)*1000:>6.1f}ms | {(t2-t0)*1000:>6.1f}ms | {fps:>5.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n📊 최종: 평균 {count/(time.time()-fps_start):.1f} FPS")


def multi_detect(pi_ip: str, front_port: int, top_port: int,
                 model_path: str, imgsz: int):
    """멀티카메라 OpenVINO YOLO 탐지 — FRONT+TOP 나란히."""
    print(f"⚡ YOLO 모델 로딩: {model_path}")
    model = YOLO(model_path, task="detect")
    print("✅ 모델 준비 완료!")

    ctx = zmq.Context()

    # FRONT 소켓
    f_sock = ctx.socket(zmq.SUB)
    f_sock.connect(f"tcp://{pi_ip}:{front_port}")
    f_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    f_sock.setsockopt(zmq.CONFLATE, 1)
    f_sock.setsockopt(zmq.RCVTIMEO, 1000)

    # TOP 소켓
    t_sock = ctx.socket(zmq.SUB)
    t_sock.connect(f"tcp://{pi_ip}:{top_port}")
    t_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    t_sock.setsockopt(zmq.CONFLATE, 1)
    t_sock.setsockopt(zmq.RCVTIMEO, 1000)

    print(f"\n🚀 OpenVINO 멀티카메라 탐지 시작!")
    print(f"📡 FRONT: tcp://{pi_ip}:{front_port}")
    print(f"📡 TOP  : tcp://{pi_ip}:{top_port}")
    print(f"{'프레임':>6} | {'FRONT':>8} | {'TOP':>8} | {'합계':>8} | {'FPS':>6}")
    print("-" * 55)

    # 대기 이미지
    waiting = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(waiting, "Waiting...", (200, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)

    count = 0
    fps_start = time.time()

    while True:
        t_all = time.time()

        # ── FRONT ──
        try:
            buf = f_sock.recv()
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            t0 = time.time()
            res = model(frame, verbose=False, imgsz=imgsz)
            f_ms = (time.time() - t0) * 1000
            ann_f = res[0].plot()
            cv2.putText(ann_f, f"FRONT {f_ms:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except zmq.Again:
            ann_f = waiting.copy()
            f_ms = 0

        # ── TOP ──
        try:
            buf = t_sock.recv()
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            t0 = time.time()
            res = model(frame, verbose=False, imgsz=imgsz)
            t_ms = (time.time() - t0) * 1000
            ann_t = res[0].plot()
            cv2.putText(ann_t, f"TOP {t_ms:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        except zmq.Again:
            ann_t = waiting.copy()
            t_ms = 0

        # ── 나란히 합쳐서 표시 (1280x480) ──
        combined = np.hstack([ann_f, ann_t])
        cv2.imshow("YOLO26 Multi-Camera (OpenVINO)", combined)
        count += 1
        total_ms = (time.time() - t_all) * 1000

        if count % 15 == 0:
            fps = count / (time.time() - fps_start)
            print(f"{count:>6} | {f_ms:>6.1f}ms | {t_ms:>6.1f}ms | {total_ms:>6.1f}ms | {fps:>5.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n📊 최종: 평균 {count/(time.time()-fps_start):.1f} FPS")


def main():
    parser = argparse.ArgumentParser(description="OpenVINO YOLO 탐지 (싱글/멀티)")
    parser.add_argument("--pi-ip", default=DEFAULT_PI_IP, help="Pi IP 주소")
    parser.add_argument("--multi", action="store_true",
                        help="멀티카메라 모드 (FRONT+TOP)")
    parser.add_argument("--front-port", type=int, default=5556)
    parser.add_argument("--top-port", type=int, default=5557)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="OpenVINO 모델 경로")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="추론 해상도 (변환 크기와 일치!)")
    args = parser.parse_args()

    if args.multi:
        multi_detect(args.pi_ip, args.front_port, args.top_port,
                     args.model, args.imgsz)
    else:
        single_detect(args.pi_ip, args.front_port, args.model, args.imgsz)


if __name__ == "__main__":
    main()
