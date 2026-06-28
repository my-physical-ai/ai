# 누크 카메라 미리보기 서버 — AI 없이 카메라 영상만 웹으로 보여주는 최소 테스트 서버
# [2026-06-28 작성] 재고조사 본 시스템 전에, 카메라가 브라우저에 잘 나오는지 확인하는 용도

import time
import threading

import cv2                                          # 카메라 영상 처리
from flask import Flask, Response                   # 웹으로 영상 스트리밍

# ============================================================
# ★ 본인 환경에 맞게 수정
# ============================================================
CAM_INDEX = 0          # ← camera_check.py로 찾은 카메라 번호
CAM_WIDTH = 1280       # ← 가로 해상도
CAM_HEIGHT = 720       # ← 세로 해상도
FLASK_PORT = 5000      # ← 웹 포트

app = Flask(__name__)
latest_frame = None                                 # 최신 카메라 프레임
frame_lock = threading.Lock()                       # 프레임 동기화


def camera_thread():
    """백그라운드에서 카메라 프레임을 계속 읽는 스레드."""
    global latest_frame
    cap = cv2.VideoCapture(CAM_INDEX)               # 카메라 열기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)    # 가로 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT) # 세로 해상도 설정
    print(f"📷 카메라 {CAM_INDEX}번 시작 ({CAM_WIDTH}x{CAM_HEIGHT})")
    while True:
        ok, frame = cap.read()                      # 프레임 읽기
        if ok:
            with frame_lock:
                latest_frame = frame                # 최신 프레임 갱신
        else:
            time.sleep(0.1)                         # 실패 시 잠깐 대기


def generate():
    """카메라 프레임을 MJPEG 스트림으로 만들어 브라우저에 보낸다."""
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            # 화면에 해상도 정보 표시 (확인용)
            h, w = frame.shape[:2]
            cv2.putText(frame, f"{w}x{h} OK", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', frame)   # JPEG 인코딩
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)                            # 약 20fps


@app.route('/')
def index():
    """카메라 영상을 보여주는 간단한 HTML 페이지."""
    return """
    <html><head><meta charset="utf-8"><title>카메라 미리보기</title></head>
    <body style="background:#0f172a;color:#e2e8f0;font-family:sans-serif;text-align:center;padding:20px;">
      <h2>📷 누크 카메라 미리보기</h2>
      <p>초록색 글씨로 해상도가 보이면 정상입니다.</p>
      <img src="/video" style="max-width:90%;border-radius:12px;border:2px solid #2563EB;">
    </body></html>
    """


@app.route('/video')
def video():
    """MJPEG 스트림 엔드포인트."""
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("=" * 52)
    print("📷 누크 카메라 미리보기 서버")
    print("=" * 52)
    # 카메라 스레드 시작 (데몬 = 메인 종료 시 함께 종료)
    threading.Thread(target=camera_thread, daemon=True).start()
    print(f"🌐 브라우저에서 http://누크IP:{FLASK_PORT} 접속")
    print("=" * 52)
    app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)   # 서버 시작
