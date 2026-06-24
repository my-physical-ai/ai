# test_camera.py
# 본 프로그램 실행 전, RealSense D435i가 정상 동작하는지 확인하는 테스트 스크립트

import pyrealsense2 as rs   # RealSense SDK
import numpy as np          # 수치 배열

print("🔍 RealSense 연결 테스트 시작...")

pipeline = rs.pipeline()    # 파이프라인 생성
config = rs.config()        # 설정 객체
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 컬러 스트림
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # depth 스트림

try:
    profile = pipeline.start(config)   # 시작 시도
    scale = profile.get_device().first_depth_sensor().get_depth_scale()  # 환산 계수
    print(f"✅ 카메라 연결 성공 | depth_scale = {scale}")

    align = rs.align(rs.stream.color)  # 정렬 객체
    for i in range(5):
        frames = pipeline.wait_for_frames()       # 프레임 대기
        aligned = align.process(frames)           # 정렬
        color = np.asanyarray(aligned.get_color_frame().get_data())  # 컬러
        depth = np.asanyarray(aligned.get_depth_frame().get_data())  # depth
        cy, cx = depth.shape[0] // 2, depth.shape[1] // 2  # 화면 중앙
        dist_m = depth[cy, cx] * scale            # 중앙 거리(m)
        print(f"  프레임{i+1}: 컬러 {color.shape}, 중앙거리 {dist_m:.3f}m")

    print("✅ 테스트 완료 — 카메라 정상! app.py를 실행하세요.")
except Exception as e:
    print(f"❌ 카메라 오류: {e}")
    print("→ USB3 연결 확인, 'rs-enumerate-devices'로 인식 여부 점검")
finally:
    pipeline.stop()        # 파이프라인 정리
