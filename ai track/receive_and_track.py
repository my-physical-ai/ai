# Pi 듀얼 카메라 수신 → YOLO26 OpenVINO + BoT-SORT/ByteTrack 트래킹 (NUC)
# [2026-02-21 추가] 듀얼 카메라(FRONT 5556 + TOP 5557) 동시 수신 + 트래커 전환
#
# ★ 왜 NUC에서 실행하는가?
#   Pi 5 (ARM CPU):  PyTorch YOLO26n ≈ 500ms/프레임 (2fps) → 실시간 불가
#                     OpenVINO 사용 불가 (Intel CPU 전용)
#   NUC i7 (Intel):  OpenVINO YOLO26n ≈ 35ms/프레임 (28fps) → 실시간 OK!
#                     BoT-SORT CMC도 Intel CPU에서 최적 성능
#
# ★ Pi는 뭘 하나?
#   카메라 캡처(가벼움) + JPEG 압축 + ZeroMQ 전송 = Pi에 적합한 작업
#   send_camera_front.py → 포트 5556 (FRONT 카메라)
#   send_camera_top.py   → 포트 5557 (TOP 카메라)
#
# 실행:
#   Pi:  터미널1) python send_camera_front.py
#        터미널2) python send_camera_top.py
#   NUC: conda activate lerobot && python receive_and_track.py
#
# 키보드 (영상 창 포커스 필요!):
#   q=종료, t=트래커전환(BoT-SORT↔ByteTrack), c=카메라전환(FRONT↔TOP↔DUAL)
#   p=사람필터, s=스크린샷

import cv2
import zmq
import numpy as np
import time
import os
from collections import defaultdict

from ultralytics import YOLO

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PI_IP = "192.168.50.111"                # Pi5 IP (hostname -I로 확인)
FRONT_PORT = 5556                       # FRONT 카메라 포트
TOP_PORT = 5557                         # TOP 카메라 포트
YOLO_MODEL = "/home/zetabank/lerobot/yolo26n_openvino_model/"  # OpenVINO 절대경로
IMGSZ = 640                             # OpenVINO 변환 크기와 반드시 일치
CONFIDENCE = 0.3                        # ByteTrack은 낮은 신뢰도도 2차 매칭에 활용
TRAIL_LEN = 30                          # 궤적 길이 (프레임 수)

# [2026-02-21 추가] BoT-SORT 커스텀 설정 파일 경로 (스크립트와 같은 폴더)
BOTSORT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort_lekiwi.yaml")
BYTETRACK_YAML = "bytetrack.yaml"       # ultralytics 내장 YAML은 문자열만으로 OK

# [2026-02-21 추가] 카메라 뷰 모드
VIEW_FRONT = 0     # FRONT 카메라만 (트래킹 적용)
VIEW_TOP = 1        # TOP 카메라만 (트래킹 적용)
VIEW_DUAL = 2       # 좌: FRONT + 우: TOP 나란히 표시

# COCO 클래스 이름 (자주 등장하는 것만)
COCO = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog",
    24: "backpack", 39: "bottle", 56: "chair", 62: "tv",
    63: "laptop", 67: "phone", 73: "book",
}


def create_botsort_yaml(path: str) -> None:
    """LeKiwi 이동 로봇 최적화된 BoT-SORT 설정 파일을 생성한다.

    Args:
        path: 설정 파일 저장 경로
    """
    # [2026-02-16 추가] LeKiwi 이동 로봇용 BoT-SORT 설정
    # [2026-02-21 수정] gmc_method를 orb로 변경 (NUC CPU 최적화)
    yaml_content = """# BoT-SORT: LeKiwi 이동 로봇 최적화 설정
# ByteTrack 기반 + 카메라 모션 보상 + 선택적 Re-ID
tracker_type: botsort

# ── 매칭 임계값 ──
track_high_thresh: 0.25    # 1차 매칭: 확실한 탐지
track_low_thresh: 0.1      # 2차 매칭: 가려진 객체 복구
new_track_thresh: 0.25     # 새 트랙 생성 기준
track_buffer: 30           # Lost 유지 프레임 (15fps에서 2초)
match_thresh: 0.8          # IoU 매칭 임계값

# ── BoT-SORT 고유 기능 ──
fuse_score: true           # 탐지 신뢰도를 매칭에 반영
gmc_method: orb            # 카메라 모션 보상 방법 (LeKiwi 이동 시 핵심!)
                           # orb: ORB 특징점 기반 (NUC CPU에서 빠름 ⭐)
                           # sparseOptFlow: 광학흐름 기반 (정확하지만 느림)
                           # none: 비활성화 (고정 카메라용)

# ── Re-ID (선택적) ──
with_reid: false           # OpenVINO 모델에서는 false 필수 (특징 추출 제한)
proximity_thresh: 0.5      # Re-ID 활성화 시: 근접 임계값
appearance_thresh: 0.25    # Re-ID 활성화 시: 외형 유사도 임계값
"""
    with open(path, 'w') as f:
        f.write(yaml_content)
    print(f"[NUC] BoT-SORT 설정 생성: {path}")


def create_zmq_subscriber(pi_ip: str, port: int) -> zmq.Socket:
    """ZeroMQ SUB 소켓을 생성하고 Pi에 연결한다.

    Args:
        pi_ip: Pi IP 주소
        port: ZeroMQ 포트

    Returns:
        설정 완료된 ZeroMQ SUB 소켓
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{pi_ip}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 100)     # 100ms 타임아웃 (빠른 카메라 전환용)
    sock.setsockopt(zmq.CONFLATE, 1)        # 최신 프레임만 유지 (단일 JPEG 전송이므로 OK)
    return sock


def recv_frame(sock: zmq.Socket) -> np.ndarray | None:
    """ZeroMQ 소켓에서 JPEG 프레임을 수신하여 BGR 배열로 디코딩한다.

    Args:
        sock: ZeroMQ SUB 소켓

    Returns:
        BGR numpy 배열 또는 수신 실패 시 None
    """
    try:
        buf = sock.recv()                                           # JPEG 바이트 수신
        frame = cv2.imdecode(np.frombuffer(buf, np.uint8), 1)       # JPEG → BGR
        return frame
    except zmq.Again:
        return None                                                 # 타임아웃


def color_for(tid: int) -> tuple:
    """트랙 ID별 고유 BGR 색상을 반환한다.

    Args:
        tid: 트랙 고유 ID

    Returns:
        BGR 색상 튜플
    """
    np.random.seed(tid * 7 + 13)
    return tuple(np.random.randint(80, 255, 3).tolist())


def draw_trail(frame: np.ndarray, pts: list, color: tuple) -> None:
    """궤적을 점점 굵은 선으로 그린다.

    Args:
        frame: 그릴 프레임
        pts: 중심점 좌표 리스트
        color: BGR 색상
    """
    for i in range(1, len(pts)):
        th = int(np.sqrt(i / 2.0) * 2) + 1                         # 점점 두꺼워지는 선
        cv2.line(frame, pts[i - 1], pts[i], color, th)


def draw_tracking(frame: np.ndarray, results, trails: dict,
                  all_ids: set, person_only: bool) -> int:
    """YOLO 트래킹 결과를 프레임에 시각화한다.

    Args:
        frame: 그릴 프레임 (in-place 수정)
        results: model.track() 결과
        trails: 궤적 딕셔너리 {tid: [(x,y), ...]}
        all_ids: 전체 고유 ID 집합
        person_only: True면 사람만 표시

    Returns:
        현재 프레임에서 활성 객체 수
    """
    active = 0
    if results[0].boxes.id is None:
        return active

    boxes = results[0].boxes
    ids = boxes.id.int().cpu().tolist()                             # 트랙 ID
    xyxys = boxes.xyxy.cpu().numpy()                                # 바운딩 박스
    clss = boxes.cls.int().cpu().tolist()                           # 클래스 번호
    confs = boxes.conf.cpu().tolist()                               # 신뢰도

    for tid, xyxy, cls, conf in zip(ids, xyxys, clss, confs):
        if person_only and cls != 0:                                # 사람 필터
            continue

        active += 1
        all_ids.add(tid)
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        c = color_for(tid)

        # 궤적 그리기
        trails[tid].append((cx, cy))
        trails[tid] = trails[tid][-TRAIL_LEN:]                      # 최대 길이 제한
        if len(trails[tid]) > 1:
            draw_trail(frame, trails[tid], c)

        # 바운딩 박스 + 라벨
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        nm = COCO.get(cls, f"cls:{cls}")
        lb = f"ID:{tid} {nm} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(lb, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, lb, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame, (cx, cy), 3, c, -1)

    return active


def make_placeholder(text: str = "Waiting...") -> np.ndarray:
    """카메라 미연결 시 표시할 대기 이미지를 생성한다.

    Args:
        text: 표시할 텍스트

    Returns:
        480x640 BGR 프레임
    """
    ph = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(ph, text, (140, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    return ph


def main() -> None:
    """Pi 듀얼 카메라 수신 → YOLO26 + BoT-SORT/ByteTrack 트래킹 메인 루프."""

    # ── BoT-SORT 설정 파일 생성 ──
    if not os.path.exists(BOTSORT_YAML):
        create_botsort_yaml(BOTSORT_YAML)

    # ── ZeroMQ 소켓 2개 (FRONT + TOP) ──
    print(f"[NUC] Pi({PI_IP}) 연결 중...")
    sock_front = create_zmq_subscriber(PI_IP, FRONT_PORT)           # FRONT 카메라
    sock_top = create_zmq_subscriber(PI_IP, TOP_PORT)               # TOP 카메라
    print(f"[NUC] FRONT→포트{FRONT_PORT}, TOP→포트{TOP_PORT} 준비 완료")

    # ── YOLO 모델 로딩 ──
    print(f"[NUC] 모델 로딩: {YOLO_MODEL} (imgsz={IMGSZ})")
    model = YOLO(YOLO_MODEL, task="detect")

    # ── 트래커 상태 ──
    use_botsort = True                                              # 기본: BoT-SORT
    current_tracker = BOTSORT_YAML
    person_only = False

    # ── 카메라 뷰 상태 ──
    view_mode = VIEW_FRONT                                          # 기본: FRONT 카메라
    view_names = {VIEW_FRONT: "FRONT", VIEW_TOP: "TOP", VIEW_DUAL: "DUAL"}

    # ── 시각화 상태 ──
    trails = defaultdict(list)
    all_ids = set()
    prev = time.time()

    tracker_name = "BoT-SORT" if use_botsort else "ByteTrack"
    print(f"\n[NUC] 트래킹 시작! 트래커: {tracker_name}, 뷰: {view_names[view_mode]}")
    print("[NUC] 키: q=종료, t=트래커전환, c=카메라전환, p=사람필터, s=스크린샷")
    print("=" * 65)

    while True:
        # ── 프레임 수신 ──
        frame_front = recv_frame(sock_front)                        # FRONT 카메라
        frame_top = recv_frame(sock_top)                            # TOP 카메라

        # ── FPS 계산 ──
        now = time.time()
        fps = 1.0 / (now - prev) if now != prev else 0
        prev = now

        # ── 뷰 모드에 따라 표시할 프레임 결정 ──
        if view_mode == VIEW_FRONT:
            # FRONT 단독 — 트래킹 적용
            frame = frame_front if frame_front is not None else make_placeholder("FRONT: No signal")
            t0 = time.time()
            results = model.track(frame, persist=True, conf=CONFIDENCE,
                                  imgsz=IMGSZ, tracker=current_tracker, verbose=False)
            track_ms = (time.time() - t0) * 1000
            active = draw_tracking(frame, results, trails, all_ids, person_only)

        elif view_mode == VIEW_TOP:
            # TOP 단독 — 트래킹 적용
            frame = frame_top if frame_top is not None else make_placeholder("TOP: No signal")
            t0 = time.time()
            results = model.track(frame, persist=True, conf=CONFIDENCE,
                                  imgsz=IMGSZ, tracker=current_tracker, verbose=False)
            track_ms = (time.time() - t0) * 1000
            active = draw_tracking(frame, results, trails, all_ids, person_only)

        else:
            # DUAL — 양쪽 모두 표시 (FRONT에만 트래킹, TOP은 원본)
            f_front = frame_front if frame_front is not None else make_placeholder("FRONT: No signal")
            f_top = frame_top if frame_top is not None else make_placeholder("TOP: No signal")

            # FRONT에 트래킹 적용
            t0 = time.time()
            results = model.track(f_front, persist=True, conf=CONFIDENCE,
                                  imgsz=IMGSZ, tracker=current_tracker, verbose=False)
            track_ms = (time.time() - t0) * 1000
            active = draw_tracking(f_front, results, trails, all_ids, person_only)

            # TOP에 "TOP CAM" 라벨
            cv2.putText(f_top, "TOP CAM (raw)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

            # 좌우 나란히 합치기
            frame = np.hstack([f_front, f_top])                     # 1280x480

        # ── HUD 오버레이 ──
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 44), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        tracker_name = "BoT-SORT" if use_botsort else "ByteTrack"
        mode_str = "Person" if person_only else "All"
        cam_str = view_names[view_mode]
        hud = (f"{tracker_name} | {cam_str} | FPS:{fps:.1f} | Track:{track_ms:.0f}ms | "
               f"Obj:{active} | IDs:{len(all_ids)} | {mode_str}")
        cv2.putText(frame, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── 화면 표시 ──
        cv2.imshow("LeKiwi Dual-Cam Tracker (Pi -> NUC)", frame)

        # ── 키 입력 (영상 창 포커스 필요!) ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # [2026-02-21 추가] t키: 트래커 전환 (BoT-SORT ↔ ByteTrack)
        elif key == ord('t'):
            use_botsort = not use_botsort
            current_tracker = BOTSORT_YAML if use_botsort else BYTETRACK_YAML
            tracker_name = "BoT-SORT" if use_botsort else "ByteTrack"
            # ★ 트래커 전환 시 3가지 상태 초기화 필수! (이슈 14)
            trails.clear()                                          # 1. 궤적 초기화
            all_ids.clear()                                         # 2. ID 목록 초기화
            model.predictor = None                                  # 3. 내부 상태 리셋 (핵심!)
            print(f"[NUC] 🔄 트래커 전환: {tracker_name}")

        # [2026-02-21 추가] c키: 카메라 뷰 순환 (FRONT → TOP → DUAL → FRONT)
        elif key == ord('c'):
            view_mode = (view_mode + 1) % 3
            # 카메라 전환 시에도 트래커 상태 리셋 (다른 영상이므로)
            trails.clear()
            all_ids.clear()
            model.predictor = None
            print(f"[NUC] 📷 카메라 전환: {view_names[view_mode]}")

        elif key == ord('p'):
            person_only = not person_only
            print(f"[NUC] 사람 필터: {'ON' if person_only else 'OFF'}")

        elif key == ord('s'):
            fn = f"track_{tracker_name}_{view_names[view_mode]}_{int(time.time())}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[NUC] 📸 저장: {fn}")

    cv2.destroyAllWindows()
    print(f"\n[NUC] 종료. 총 {len(all_ids)}개 고유 ID 추적됨.")


if __name__ == "__main__":
    main()
