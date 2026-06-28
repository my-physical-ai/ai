# ============================================================
# 파일명: test_thermal_camera.py
# 설명: flirone 카메라(video2 + video3) 둘 다 확인하는 테스트
# 용도: app.py 켜기 전에 어느 장치에 흑백/컬러가 나오는지 눈으로 확인
# 사용: python test_thermal_camera.py          (브라우저로 두 영상 나란히 보기)
#       python test_thermal_camera.py --check  (터미널에서 양쪽 빠른 진단)
# ============================================================

import sys                                              # 명령행 인자
import time                                             # 시간 측정
import argparse                                         # 옵션 파싱
import threading                                        # 두 카메라 동시 처리

import cv2                                              # 영상 캡처
import numpy as np                                      # 배열 연산
from flask import Flask, Response                       # 브라우저 스트리밍

DEVICES = [2, 3]                                        # 확인할 장치 (video2, video3 둘 다)
FLASK_PORT = 5050                                       # 테스트 전용 포트 (app.py 5000과 겹치지 않게)


def quick_check(dev_index):
    """장치 하나가 열리고 프레임이 들어오는지 진단하고, 흑백/컬러를 판별한다."""
    cap = cv2.VideoCapture(dev_index)                  # 카메라 열기
    if not cap.isOpened():                             # 못 열면
        print(f"  ❌ /dev/video{dev_index} 열기 실패")
        cap.release()
        return False

    ok_count = 0                                        # 성공 프레임 수
    shape = None                                        # 프레임 크기
    is_color = None                                     # 컬러 여부
    for i in range(5):                                 # 5번 시도
        ret, frame = cap.read()                        # 한 프레임 읽기
        if ret and frame is not None:                  # 성공하면
            ok_count += 1                              # 카운트
            shape = frame.shape                        # 크기 기록
            is_color = detect_color(frame)             # 흑백/컬러 판별
        time.sleep(0.1)                                # 0.1초 간격
    cap.release()                                       # 카메라 해제

    if ok_count == 0:                                  # 한 장도 못 읽으면
        print(f"  ⚠️ /dev/video{dev_index} 열렸으나 영상 없음")
        return False

    kind = "🎨 컬러" if is_color else "⬜ 흑백(회색조)"  # 종류 표시
    print(f"  ✅ /dev/video{dev_index}: {ok_count}/5 수신, 크기 {shape[1]}x{shape[0]}, {kind}")
    return True


def detect_color(frame):
    """프레임이 컬러인지 흑백인지 판별한다 (R/G/B 채널 차이로)."""
    if frame.ndim < 3 or frame.shape[2] < 3:           # 단일 채널이면
        return False                                    # 흑백
    b, g, r = cv2.split(frame)                         # 채널 분리
    # R=G=B이면 흑백, 채널 간 차이가 크면 컬러
    diff = int(np.mean(np.abs(r.astype(int) - g.astype(int)))  # R-G 평균 차이
               + np.mean(np.abs(g.astype(int) - b.astype(int))))  # G-B 평균 차이
    return diff > 8                                    # 차이가 크면 컬러로 판정


def check_all():
    """video2, video3 둘 다 진단한다 (--check 모드)."""
    print("=" * 55)
    print("🔍 열화상 카메라 진단 — video2 + video3 둘 다")
    print("=" * 55)
    results = {}                                        # 장치별 결과
    for dev in DEVICES:                                # 각 장치
        results[dev] = quick_check(dev)                # 진단
    print("-" * 55)

    ok_devices = [d for d, ok in results.items() if ok]  # 성공한 장치
    if not ok_devices:                                 # 둘 다 실패
        print("❌ 두 장치 모두 영상 없음")
        print("   → flirone을 먼저 켜세요:")
        print("     cd ~/flirone-v4l2 && sudo ./flirone ./palettes/Grayscale.raw")
        return False
    print(f"👍 정상 장치: {', '.join(f'video{d}' for d in ok_devices)}")
    print("   브라우저로 직접 보려면: python test_thermal_camera.py")
    return True


# ============================================================
# Flask 앱 (두 영상 나란히 스트리밍)
# ============================================================
app = Flask(__name__)


def generate_stream(dev_index):
    """한 장치의 프레임을 MJPEG 스트림으로 보낸다 (장치별 독립 스트림)."""
    cap = cv2.VideoCapture(dev_index)                  # 카메라 열기
    while True:                                         # 계속 송출
        ret, frame = cap.read()                        # 프레임 읽기
        if not ret or frame is None:                   # 실패 시
            frame = np.zeros((128, 160, 3), np.uint8)  # 검은 화면
            cv2.putText(frame, "NO SIGNAL", (15, 70),  # 안내
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # 작은 영상을 보기 좋게 4배 확대
        frame = cv2.resize(frame, (640, 512), interpolation=cv2.INTER_NEAREST)  # 확대
        # 흑백/컬러 판별해서 라벨에 표시
        kind = "COLOR" if detect_color(frame) else "GRAY"  # 종류
        cv2.putText(frame, f"/dev/video{dev_index} ({kind})", (10, 28),  # 라벨
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _, jpeg = cv2.imencode(".jpg", frame)          # JPEG 인코딩
        yield (b"--frame\r\n"                           # MJPEG 전송
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.05)                               # ~20fps


@app.route("/")
def index():
    """video2, video3을 나란히 보여주는 HTML 페이지."""
    return """
    <html><head><meta charset="utf-8"><title>열화상 카메라 테스트</title>
    <style>
      body{background:#0b0e14;color:#e6edf3;font-family:sans-serif;text-align:center;padding:24px}
      h2{color:#a855f7}
      .row{display:flex;gap:20px;justify-content:center;flex-wrap:wrap;margin-top:20px}
      .cam{background:#141925;border-radius:12px;padding:14px;border:1px solid #232b3a}
      .cam h3{margin:0 0 10px;font-size:15px;color:#7dd3fc}
      img{border-radius:8px;border:2px solid #7C3AED;display:block}
      .hint{color:#64748b;margin-top:24px;font-size:14px;line-height:1.7}
    </style></head>
    <body>
      <h2>🌡️ 열화상 카메라 테스트 — video2 & video3</h2>
      <p>두 장치를 동시에 보여줍니다. 어느 쪽이 흑백(열화상)이고 어느 쪽이 컬러인지 확인하세요.</p>
      <div class="row">
        <div class="cam"><h3>📷 /dev/video2</h3><img src="/feed/2" width="480"></div>
        <div class="cam"><h3>📷 /dev/video3</h3><img src="/feed/3" width="480"></div>
      </div>
      <div class="hint">
        영상 위 라벨에 GRAY(흑백) / COLOR(컬러)가 표시됩니다.<br>
        둘 다 NO SIGNAL이면 → flirone을 먼저 켜세요.
      </div>
    </body></html>
    """


@app.route("/feed/<int:dev_index>")
def feed(dev_index):
    """장치별 MJPEG 스트림 엔드포인트 (video2 또는 video3)."""
    return Response(generate_stream(dev_index),        # 해당 장치 스트림
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="열화상 카메라 테스트 (video2+video3)")  # 파서
    parser.add_argument("--check", action="store_true",  # --check 옵션
                        help="터미널에서 양쪽 빠른 진단만 (브라우저 없이)")
    args = parser.parse_args()                         # 인자 파싱

    if args.check:                                     # --check 모드
        ok = check_all()                               # 양쪽 진단
        sys.exit(0 if ok else 1)                       # 결과 코드 반환
    else:                                              # 브라우저 모드
        print("브라우저 모드로 시작합니다. 먼저 양쪽 카메라를 확인합니다...\n")
        check_all()                                    # 진단 출력
        print(f"\n🌐 브라우저에서 확인: http://NUC_IP:{FLASK_PORT}")
        print(f"   (이 PC에서는 http://localhost:{FLASK_PORT})")
        print("   video2와 video3이 나란히 표시됩니다. 종료는 Ctrl+C\n")
        app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True)  # 서버 시작 (threaded 필수)
