# realsense_camera.py
# NUC에 연결된 RealSense D435i에서 RGB+Depth를 동시 취득하는 모듈 (생수병 검사용)

import pyrealsense2 as rs  # RealSense SDK
import numpy as np          # 수치 배열 처리
import threading            # 멀티스레드 동기화


class RealSenseCamera:
    """D435i에서 정렬된 RGB+Depth 프레임을 제공하는 클래스."""

    def __init__(self, width=848, height=480, fps=30):
        # ★ D435i 최적 depth 해상도는 848x480 (그 이상은 품질 향상 없음)
        self.width = width                    # 프레임 너비 (848 권장)
        self.height = height                  # 프레임 높이 (480 권장)
        self.fps = fps                        # 초당 프레임 수
        self.pipeline = None                  # RealSense 파이프라인
        self.align = None                     # Depth→Color 정렬 객체
        self.depth_scale = 0.001              # depth 원시값→미터 환산 계수 (기본 1mm)
        self.lock = threading.Lock()          # 프레임 접근 동기화 락
        self.color_image = None               # 최신 컬러 프레임 (BGR)
        self.depth_array = None               # 최신 depth 프레임 (uint16)
        self.running = False                  # 캡처 동작 여부

        # RealSense 후처리 필터 (노이즈 제거 — 찌그러짐 측정 정확도 핵심)
        self.spatial = rs.spatial_filter()        # 공간 평활 필터
        self.spatial.set_option(rs.option.filter_magnitude, 2)       # 필터 강도
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # 공간 평활 정도
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)   # 에지 보존 임계
        self.temporal = rs.temporal_filter()      # 시간 평활 필터
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4) # 시간 평활 정도
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)  # 변화 임계
        self.hole_filling = rs.hole_filling_filter()  # 빈 영역 채우기 (투명/반사 보완)

    def start(self):
        """카메라 파이프라인을 시작하고 캡처 스레드를 띄운다."""
        self.pipeline = rs.pipeline()         # 파이프라인 생성
        config = rs.config()                  # 스트림 설정 객체
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # 컬러
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)   # depth

        profile = self.pipeline.start(config)  # 파이프라인 시작
        depth_sensor = profile.get_device().first_depth_sensor()  # depth 센서 핸들
        self.depth_scale = depth_sensor.get_depth_scale()         # 실제 환산 계수 읽기
        print(f"✅ RealSense 시작 | depth_scale={self.depth_scale}")

        self.align = rs.align(rs.stream.color)  # Depth를 Color 좌표계에 정렬 (필수)
        self.running = True                     # 동작 플래그 ON
        threading.Thread(target=self._capture_loop, daemon=True).start()  # 백그라운드 캡처

    def _capture_loop(self):
        """백그라운드에서 계속 프레임을 받아 최신값으로 갱신하는 루프."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()  # 프레임 세트 대기
                aligned = self.align.process(frames)       # Depth를 Color에 정렬
                color_frame = aligned.get_color_frame()    # 정렬된 컬러 프레임
                depth_frame = aligned.get_depth_frame()    # 정렬된 depth 프레임
                if not color_frame or not depth_frame:
                    continue                                # 프레임 누락 시 건너뜀

                depth_frame = self.spatial.process(depth_frame)      # 공간 필터
                depth_frame = self.temporal.process(depth_frame)     # 시간 필터
                depth_frame = self.hole_filling.process(depth_frame) # 빈 영역 채우기

                color = np.asanyarray(color_frame.get_data())   # BGR 이미지
                depth = np.asanyarray(depth_frame.get_data())   # uint16 depth

                with self.lock:                  # 락 안에서 최신 프레임 교체
                    self.color_image = color
                    self.depth_array = depth
            except Exception as e:
                print(f"⚠️ 캡처 오류: {e}")       # 오류 시 루프 유지

    def get_frames(self):
        """최신 RGB+Depth 프레임의 복사본을 반환한다 (멀티스레드 안전)."""
        with self.lock:                          # 락 안에서 copy (경합 방지)
            color = self.color_image.copy() if self.color_image is not None else None
            depth = self.depth_array.copy() if self.depth_array is not None else None
        return color, depth                      # 복사본 반환

    def stop(self):
        """캡처를 멈추고 파이프라인을 정리한다."""
        self.running = False                     # 동작 플래그 OFF
        if self.pipeline:
            self.pipeline.stop()                 # 파이프라인 종료
        print("🛑 RealSense 정지")
