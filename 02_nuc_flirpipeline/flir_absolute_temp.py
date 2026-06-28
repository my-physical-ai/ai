# ============================================================
# 파일명: flir_absolute_temp.py
# 설명: FLIR ONE 앱으로 찍은 JPEG에서 절대 온도(℃)를 추출 — PyTorch Planck 변환
# 원리: 드라이버가 버린 raw 16비트 + 카메라 고유 Planck 상수가 JPEG에 내장됨
# 사전준비: sudo apt install libimage-exiftool-perl imagemagick
#          pip install flirimageextractor --break-system-packages
# 사용: python flir_absolute_temp.py photo.jpg
# ============================================================

import sys                                              # 명령행 인자
import subprocess                                       # exiftool 호출
import numpy as np                                      # 배열 연산
import torch                                            # Planck 변환 (GPU 가속)


def read_planck_constants(jpg_path):
    """exiftool로 해당 JPEG에서 카메라 고유 Planck 상수 5개를 읽는다."""
    keys = ["PlanckR1", "PlanckR2", "PlanckB", "PlanckF", "PlanckO"]  # 보정 상수 이름
    consts = {}                                          # 상수 저장 딕셔너리
    for k in keys:                                        # 각 상수 추출
        out = subprocess.check_output(                   # exiftool 단일 태그 질의
            ["exiftool", "-b", f"-{k}", jpg_path], text=True).strip()
        consts[k] = float(out)                            # 실수 변환 저장
    return consts                                         # 카메라 고유 상수 반환


def read_raw_thermal(jpg_path):
    """exiftool + imagemagick으로 raw 16비트 thermal 배열을 추출한다."""
    # JPEG 내장 RawThermalImage를 16비트 PNG로 추출 (FLIR ONE은 160x120)
    raw_png = subprocess.check_output(                   # raw 바이너리 → png 파이프
        f"exiftool -b -RawThermalImage '{jpg_path}' | "
        f"convert - gray:- | convert -depth 16 -endian lsb -size 160x120 gray:- png:-",
        shell=True)
    # png 바이트를 numpy로 디코딩
    import cv2                                            # 이미지 디코딩
    arr = cv2.imdecode(np.frombuffer(raw_png, np.uint8), cv2.IMREAD_UNCHANGED)  # 16비트 디코딩
    return arr.astype(np.float32)                         # raw 신호 배열 반환


def raw2temp_torch(raw, c, E=0.95, OD=1.0, RTemp=20.0, device="cpu"):
    """PyTorch로 raw 신호를 방사율 보정된 절대 온도(℃)로 변환한다."""
    raw_t = torch.from_numpy(raw).to(device)             # raw → 텐서
    PR1, PR2 = c["PlanckR1"], c["PlanckR2"]              # Planck R1, R2
    PB, PF, PO = c["PlanckB"], c["PlanckF"], c["PlanckO"]  # Planck B, F, O
    # 주변 반사 성분 계산 (반사된 복사를 제거하기 위함)
    raw_refl = PR1 / (PR2 * (np.exp(PB / (RTemp + 273.15)) - PF)) - PO  # 반사 raw
    raw_obj = (raw_t - (1.0 - E) * raw_refl) / E         # 방사율 보정된 물체 raw
    # Planck 역식으로 켈빈 온도 계산
    temp_k = PB / torch.log(PR1 / (PR2 * (raw_obj + PO)) + PF)  # 켈빈
    temp_c = temp_k - 273.15                              # 섭씨 변환
    return temp_c.cpu().numpy()                           # 픽셀별 절대 온도(℃) 배열


def measure_3zone(temp_c, box=None):
    """절대 온도 배열을 상/중/하 3등분하여 각 구역 평균 ℃를 측정한다."""
    h, w = temp_c.shape                                  # 배열 크기
    if box is None:                                       # 박스 없으면 전체 사용
        x1, y1, x2, y2 = 0, 0, w, h
    else:                                                 # 박스 있으면 영역 제한
        x1, y1, x2, y2 = box
    third = (y2 - y1) // 3                                # 3등분 높이
    names = ["상", "중", "하"]                            # 구역 이름
    result = []                                           # 구역별 결과
    for i, nm in enumerate(names):                        # 각 구역
        zy1 = y1 + third * i                              # 구역 상단
        zy2 = y1 + third * (i + 1) if i < 2 else y2       # 구역 하단
        region = temp_c[zy1:zy2, x1:x2]                   # 구역 온도 영역
        # 따뜻한 상위 30%만 인물로 보고 평균 (차가운 배경 제외)
        warm = region[region >= np.percentile(region, 70)]  # 따뜻한 픽셀
        result.append({"name": nm, "mean_c": round(float(warm.mean()), 1),  # 평균 ℃
                       "max_c": round(float(region.max()), 1)})            # 최고 ℃
    return result                                         # 절대온도 3구역 결과 반환


if __name__ == "__main__":
    if len(sys.argv) < 2:                                 # 인자 확인
        print("사용법: python flir_absolute_temp.py photo.jpg")
        sys.exit(1)
    jpg = sys.argv[1]                                     # JPEG 경로
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 가능하면 사용

    print(f"📷 {jpg} 처리 중... (device={dev})")
    consts = read_planck_constants(jpg)                  # 카메라 고유 상수 읽기
    print(f"⚙️ Planck 상수: {consts}")
    raw = read_raw_thermal(jpg)                           # raw 16비트 추출
    temp_c = raw2temp_torch(raw, consts, device=dev)     # 절대 온도 변환
    print(f"🌡️ 전체 온도범위: {temp_c.min():.1f} ~ {temp_c.max():.1f}℃")
    zones = measure_3zone(temp_c)                         # 3등분 측정
    print("\n=== 3등분 절대 온도 ===")
    for z in zones:                                       # 구역별 출력
        print(f"  {z['name']}: 평균 {z['mean_c']}℃ (최고 {z['max_c']}℃)")
