# test_receive.py — NUC에서 Pi 영상 수신 테스트 (1대 또는 2대 카메라)
# [2026-02-16] 싱글 모드: FRONT만 수신 / 멀티 모드: FRONT+TOP 동시 수신
"""
실행 예시:
  python3 test_receive.py                          # 싱글 (FRONT만, 포트 5556)
  python3 test_receive.py --multi                  # 멀티 (FRONT:5556 + TOP:5557)
  python3 test_receive.py --pi-ip 192.168.50.111   # IP 변경
"""

import argparse
import threading
import time
import sys

import zmq


def receive_test(name: str, pi_ip: str, port: int, count: int = 20) -> dict:
    """단일 포트에서 프레임을 수신하고 결과를 반환한다.

    Args:
        name: 카메라 이름 (로그용)
        pi_ip: Pi IP 주소
        port: ZeroMQ 포트
        count: 테스트 프레임 수

    Returns:
        결과 딕셔너리 (성공 여부, 평균 크기, 프레임 수)
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{pi_ip}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 5000)             # 5초 타임아웃

    print(f"  [{name}] 수신 대기 중... (tcp://{pi_ip}:{port})")

    total_bytes = 0
    received = 0

    for i in range(count):
        t0 = time.time()
        try:
            buf = sock.recv()
        except zmq.Again:
            print(f"  [{name}] ❌ 타임아웃! Pi에서 전송 중인지 확인하세요.")
            sock.close()
            ctx.term()
            return {"name": name, "ok": False, "count": 0}

        ms = (time.time() - t0) * 1000
        kb = len(buf) / 1024
        received += 1
        total_bytes += len(buf)

        # 5프레임마다 출력
        if received % 5 == 0:
            print(f"  [{name}] 프레임 {received:>3} | {kb:>7.1f}KB | {ms:>6.1f}ms")

    avg_kb = (total_bytes / received) / 1024 if received > 0 else 0
    sock.close()
    ctx.term()
    return {"name": name, "ok": True, "count": received, "avg_kb": avg_kb,
            "total_kb": total_bytes / 1024}


def main():
    parser = argparse.ArgumentParser(description="Pi 영상 수신 테스트")
    parser.add_argument("--pi-ip", default="192.168.50.111",
                        help="Pi IP 주소 (기본 192.168.50.111)")
    parser.add_argument("--multi", action="store_true",
                        help="멀티카메라 모드 (FRONT+TOP 동시 수신)")
    parser.add_argument("--front-port", type=int, default=5556,
                        help="FRONT 포트 (기본 5556)")
    parser.add_argument("--top-port", type=int, default=5557,
                        help="TOP 포트 (기본 5557)")
    parser.add_argument("--count", type=int, default=20,
                        help="테스트 프레임 수 (기본 20)")
    args = parser.parse_args()

    if args.multi:
        # ── 멀티카메라 모드: 2대 동시 수신 ──
        print(f"📡 멀티카메라 수신 테스트 (FRONT + TOP)")
        print(f"   Pi: {args.pi_ip}")
        print(f"   FRONT: 포트 {args.front_port} / TOP: 포트 {args.top_port}")
        print("-" * 50)

        results = [None, None]

        def test_cam(idx, name, port):
            results[idx] = receive_test(name, args.pi_ip, port, args.count)

        # 두 카메라를 동시에 수신 테스트
        t1 = threading.Thread(target=test_cam, args=(0, "FRONT", args.front_port))
        t2 = threading.Thread(target=test_cam, args=(1, "TOP", args.top_port))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 결과 요약
        print("\n" + "=" * 50)
        print("📊 멀티카메라 수신 결과")
        print("=" * 50)
        for r in results:
            if r and r["ok"]:
                print(f"  ✅ [{r['name']}] {r['count']}프레임 수신, "
                      f"평균 {r['avg_kb']:.1f}KB/프레임")
            elif r:
                print(f"  ❌ [{r['name']}] 수신 실패!")
    else:
        # ── 싱글카메라 모드: FRONT만 수신 ──
        print(f"📡 싱글카메라 수신 테스트 (FRONT)")
        print(f"   Pi: {args.pi_ip}:{args.front_port}")
        print("-" * 50)

        r = receive_test("FRONT", args.pi_ip, args.front_port, args.count)
        print("\n" + "=" * 50)
        if r["ok"]:
            print(f"✅ 수신 성공! {r['count']}프레임, 평균 {r['avg_kb']:.1f}KB")
        else:
            print("❌ 수신 실패!")
            print(f"   → Pi에서 send_camera.py 또는 multi_send_camera.py 실행 중인가요?")
            print(f"   → PI_IP={args.pi_ip} 가 맞나요? (Pi에서 hostname -I 확인)")


if __name__ == "__main__":
    main()
