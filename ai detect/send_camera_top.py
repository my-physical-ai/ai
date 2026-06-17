# send_camera_top.py — Pi TOP 카메라를 ZeroMQ 포트 5557로 전송
# [2026-02-16] 멀티카메라 구성 시 터미널 2에서 실행

import argparse
import cv2
import zmq
import time

def main():
    parser = argparse.ArgumentParser(description="TOP 카메라 전송")
    parser.add_argument("--device", default="/dev/video3", help="카메라 장치")
    parser.add_argument("--port", type=int, default=5557, help="ZeroMQ 포트")
    parser.add_argument("--fps", type=int, default=15, help="목표 FPS")
    parser.add_argument("--quality", type=int, default=80, help="JPEG 품질")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ 카메라 열기 실패: {args.device}"); return

    interval = 1.0 / args.fps
    print(f"📷 [TOP] 전송 시작! {args.device} → 포트 {args.port} ({args.fps}fps)")

    count = 0; start = time.time(); next_time = start
    while True:
        ret, frame = cap.read()
        if not ret: print("❌ 프레임 실패"); time.sleep(0.1); continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        sock.send(buf.tobytes()); count += 1
        if count % 30 == 0:
            print(f"[TOP] 전송 {count}프레임 | 실제 {count/(time.time()-start):.1f}fps")
        next_time += interval
        s = next_time - time.time()
        if s > 0: time.sleep(s)

if __name__ == "__main__":
    main()
