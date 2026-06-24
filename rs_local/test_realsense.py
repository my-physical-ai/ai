# RealSense 연결 및 기능 테스트 스크립트 — NUC에서 app.py 실행 전 사전 검증용
# [2026-02-21 생성] 6가지 항목을 순서대로 테스트하고 결과를 PASS/FAIL로 출력

import sys    # 종료 코드 반환용
import time   # 프레임 획득 시간 측정용

def main():
    """RealSense 카메라 6가지 테스트를 순서대로 실행한다."""
    print("=" * 55)
    print("🔍 RealSense 사전 테스트 (app.py 실행 전 확인)")
    print("=" * 55)

    results = []   # 테스트 결과 저장 리스트

    # --- 테스트 1: pyrealsense2 import ---
    print("\n[1/6] pyrealsense2 패키지 import ...")
    try:
        import pyrealsense2 as rs             # RealSense SDK import
        print(f"  ✅ PASS — 버전: {rs.__version__}")
        results.append(("pyrealsense2 import", True))
    except ImportError as e:
        print(f"  ❌ FAIL — {e}")
        print("  → pip install pyrealsense2 실행하세요")
        results.append(("pyrealsense2 import", False))
        show_summary(results)                 # import 실패 시 나머지 불가
        return

    # --- 테스트 2: 카메라 연결 확인 ---
    print("\n[2/6] RealSense 카메라 연결 확인 ...")
    ctx = rs.context()                        # RealSense 컨텍스트 생성
    devices = ctx.query_devices()             # 연결된 장치 목록 조회
    if len(devices) == 0:
        print("  ❌ FAIL — RealSense 카메라가 연결되지 않았습니다")
        print("  → USB 3.0 (파란색) 포트에 꽂았는지 확인하세요")
        print("  → lsusb | grep Intel 로 확인하세요")
        results.append(("카메라 연결", False))
        show_summary(results)
        return
    # 장치 정보 출력
    dev = devices[0]                          # 첫 번째 장치
    name = dev.get_info(rs.camera_info.name)          # 장치 이름
    serial = dev.get_info(rs.camera_info.serial_number)  # 시리얼 번호
    fw = dev.get_info(rs.camera_info.firmware_version)   # 펌웨어 버전
    print(f"  ✅ PASS — {name} (S/N: {serial}, FW: {fw})")
    results.append(("카메라 연결", True))

    # --- 테스트 3: 파이프라인 시작 (Color + Depth) ---
    print("\n[3/6] 파이프라인 시작 (640x480 Color+Depth) ...")
    pipeline = rs.pipeline()                  # 파이프라인 생성
    config = rs.config()                      # 스트림 설정
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)   # Color 스트림
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)    # Depth 스트림
    try:
        profile = pipeline.start(config)      # 스트림 시작
        print("  ✅ PASS — 파이프라인 시작 성공")
        results.append(("파이프라인 시작", True))
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        print("  → 다른 프로그램이 카메라를 사용 중인지 확인하세요")
        results.append(("파이프라인 시작", False))
        show_summary(results)
        return

    # --- 테스트 4: Depth 스케일 + Intrinsics 읽기 ---
    print("\n[4/6] Depth 스케일 + 카메라 파라미터 ...")
    depth_sensor = profile.get_device().first_depth_sensor()  # Depth 센서 접근
    depth_scale = depth_sensor.get_depth_scale()              # 스케일 값 읽기
    intrinsics = profile.get_stream(
        rs.stream.color).as_video_stream_profile().get_intrinsics()  # 내부 파라미터
    print(f"  ✅ PASS — depth_scale={depth_scale:.6f}")
    print(f"           fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}")
    print(f"           cx={intrinsics.ppx:.1f} cy={intrinsics.ppy:.1f}")
    results.append(("Depth 스케일/Intrinsics", True))

    # --- 테스트 5: 프레임 획득 테스트 (5프레임) ---
    print("\n[5/6] 프레임 획득 테스트 (5프레임) ...")
    import numpy as np                        # NumPy import (프레임 분석용)
    align = rs.align(rs.stream.color)         # Depth→Color 정렬
    frame_ok = 0                              # 성공 프레임 카운터
    for i in range(5):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)  # 프레임 대기
            aligned = align.process(frames)                     # 정렬 적용
            color = aligned.get_color_frame()                   # Color 프레임
            depth = aligned.get_depth_frame()                   # Depth 프레임
            if color and depth:
                c_arr = np.asanyarray(color.get_data())         # NumPy 변환
                d_arr = np.asanyarray(depth.get_data())         # NumPy 변환
                # 중심점 거리 측정
                center_dist = depth.get_distance(320, 240)
                print(f"  프레임 {i+1}: Color={c_arr.shape} "
                      f"Depth={d_arr.shape} 중심거리={center_dist:.2f}m")
                frame_ok += 1
        except Exception as e:
            print(f"  프레임 {i+1}: ❌ 실패 — {e}")

    if frame_ok == 5:
        print(f"  ✅ PASS — {frame_ok}/5 프레임 성공")
        results.append(("프레임 획득", True))
    else:
        print(f"  ⚠️ WARN — {frame_ok}/5 프레임 성공")
        results.append(("프레임 획득", frame_ok >= 3))   # 3개 이상이면 PASS

    # --- 테스트 6: 필터 체인 테스트 ---
    print("\n[6/6] Depth 필터 체인 테스트 ...")
    try:
        frames = pipeline.wait_for_frames(timeout_ms=2000)      # 프레임 획득
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()

        # 필터 생성 및 적용 (Spatial → Temporal → HoleFill)
        sf = rs.spatial_filter()                                # 공간 필터
        tf = rs.temporal_filter()                               # 시간 필터
        hf = rs.hole_filling_filter()                           # 빈 영역 채우기

        filtered = sf.process(depth_frame)                      # Spatial 적용
        filtered = tf.process(filtered)                         # Temporal 적용
        filtered = hf.process(filtered)                         # HoleFill 적용

        raw_arr = np.asanyarray(depth_frame.get_data())         # 원본 배열
        filt_arr = np.asanyarray(filtered.get_data())           # 필터 배열

        raw_zeros = np.sum(raw_arr == 0)                        # 원본 무효 픽셀 수
        filt_zeros = np.sum(filt_arr == 0)                      # 필터 무효 픽셀 수
        print(f"  RAW  무효 픽셀: {raw_zeros:,}개")
        print(f"  FILT 무효 픽셀: {filt_zeros:,}개 (감소: {raw_zeros - filt_zeros:,})")
        print("  ✅ PASS — 필터 체인 정상")
        results.append(("필터 체인", True))
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        results.append(("필터 체인", False))

    # 파이프라인 정지
    pipeline.stop()
    print("\n🛑 파이프라인 정지 완료")

    # 최종 요약
    show_summary(results)


def show_summary(results):
    """테스트 결과 요약 출력."""
    print("\n" + "=" * 55)
    print("📊 테스트 결과 요약")
    print("=" * 55)

    all_pass = True                           # 전체 통과 여부
    for name, ok in results:
        icon = "✅" if ok else "❌"            # 통과/실패 아이콘
        print(f"  {icon} {name}")
        if not ok:
            all_pass = False

    print("=" * 55)
    if all_pass:
        print("🎉 모든 테스트 통과! python3 app.py 실행 가능")
    else:
        print("⚠️ 일부 테스트 실패 — 위 오류를 해결한 후 다시 시도하세요")

    sys.exit(0 if all_pass else 1)            # 스크립트 종료 코드


if __name__ == '__main__':
    main()
