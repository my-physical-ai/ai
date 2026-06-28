# ============================================================
# 파일명: services/thermal_camera.py
# 설명: FLIR/Seek USB 열화상 카메라 — 온도 raw 배열 + 의사색상 RGB 분리 제공
# 변경 이력:
#   - [2026-06-22] 최초 생성: 백그라운드 캡처 + raw/시각화 분리
# ============================================================

import threading                                  # 캡처 스레드
import cv2                                         # 카메라 입출력
import numpy as np                                # 배열 연산


class ThermalCamera:
    """열화상 카메라 래퍼.

    ⚠️ 핵심 원칙: 온도 raw(float32 ℃)는 절대 JPEG 압축 금지 (정밀도 손실).
                YOLO/SAM3 입력용 의사색상 RGB만 별도 생성.
    """

    # 컬러맵 이름 → OpenCV 상수 매핑
    _CMAP = {"inferno": cv2.COLORMAP_INFERNO,
             "magma": cv2.COLORMAP_MAGMA,
             "jet": cv2.COLORMAP_JET,
             "hot": cv2.COLORMAP_HOT}

    def __init__(self, cfg):
        """cfg: config.yaml의 thermal 섹션."""
        self.idx = cfg["device_index"]            # USB 장치 인덱스
        self.scale = cfg["raw_scale"]             # raw→℃ 스케일
        self.offset = cfg["raw_offset"]           # raw→℃ 오프셋
        self.cmap = self._CMAP.get(cfg.get("colormap", "inferno"),
                                   cv2.COLORMAP_INFERNO)
        self.cap = None                            # VideoCapture
        self._temp_c = None                        # 최신 온도 배열
        self._vis = None                           # 최신 의사색상
        self._lock = threading.Lock()              # 프레임 동기화
        self._running = False                      # 스레드 상태

    def start(self):
        """카메라를 열고 백그라운드 캡처 스레드를 시작한다."""
        self.cap = cv2.VideoCapture(self.idx)      # USB 카메라 열기
        if not self.cap.isOpened():
            raise RuntimeError("열화상 카메라 열기 실패 — fuser /dev/video0 확인")
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()  # 캡처 루프

    def _loop(self):
        """백그라운드에서 계속 프레임을 읽어 온도/시각화로 변환."""
        while self._running:
            ret, raw = self.cap.read()             # 원본 프레임
            if not ret:
                continue

            raw16 = raw.astype(np.float32)         # float 변환
            if raw16.ndim == 3:                    # 3채널이면 1채널로
                raw16 = raw16[:, :, 0]

            temp_c = raw16 * self.scale + self.offset       # 섭씨 온도 배열
            norm = cv2.normalize(raw16, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)  # 0~255 정규화
            vis = cv2.applyColorMap(norm, self.cmap)         # 의사색상 변환

            with self._lock:                        # 최신 프레임 갱신
                self._temp_c = temp_c
                self._vis = vis

    def read(self):
        """최신 (온도배열 float32 ℃, 의사색상 BGR) 반환."""
        with self._lock:
            if self._temp_c is None:
                return None, None
            return self._temp_c.copy(), self._vis.copy()  # copy 필수 (멀티스레드)

    def is_ready(self):
        """프레임 수신 여부."""
        with self._lock:
            return self._temp_c is not None

    def stop(self):
        """캡처 중단 및 자원 해제."""
        self._running = False
        if self.cap:
            self.cap.release()
