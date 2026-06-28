# ============================================================
# 파일명: test_palette.py
# 설명: flirone의 여러 팔레트를 차례로 시험하며 비교하는 테스트 도구
# 용도: 목적별로 어떤 팔레트가 좋은지 직접 보고 고르기 (현장점검/AI학습/발표)
# 사용: python test_palette.py --list        팔레트 목록 보기
#       python test_palette.py --run Iron_Black   특정 팔레트로 flirone 실행
#       python test_palette.py --tour          추천 팔레트 4종 자동 순회 (각 10초)
# ============================================================

import os                                               # 경로/파일 확인
import sys                                               # 종료 코드
import time                                              # 대기
import argparse                                          # 옵션 파싱
import subprocess                                        # flirone 실행

FLIR_DIR = os.path.expanduser("~/flirone-v4l2")          # flirone 설치 폴더
PALETTE_DIR = os.path.join(FLIR_DIR, "palettes")         # 팔레트 폴더

# 목적별 추천 팔레트 (빅맨님 6강 노하우 기반)
RECOMMENDED = [
    ("Iron_Black", "🔥 현장 점검용", "검정→빨강→노랑→흰색. FLIR 기본 느낌, 발열 확인에 최적"),
    ("Rainbow",    "🌈 발표·시연용", "무지개색. 온도 차이가 눈에 확 띔, 시연에 좋음"),
    ("Grayscale",  "🤖 AI 학습용",  "흑백. YOLO·OpenCV·SAM 분석에 깔끔함 (app.py 권장)"),
    ("Iron2",      "📊 대비 강조용", "Iron 변형. 온도 대비를 더 강하게 표현"),
]


def list_palettes():
    """palettes 폴더의 모든 팔레트 파일을 보여준다."""
    print("=" * 58)
    print("🎨 사용 가능한 팔레트 목록")
    print("=" * 58)
    if not os.path.isdir(PALETTE_DIR):                  # 폴더 없으면
        print(f"❌ 팔레트 폴더가 없습니다: {PALETTE_DIR}")
        print("   → flirone-v4l2를 먼저 설치하세요 (4-1 단계)")
        return False

    # 실제 존재하는 .raw 팔레트 파일 수집
    files = sorted(f for f in os.listdir(PALETTE_DIR) if f.endswith(".raw"))  # raw 목록
    print(f"\n📁 {PALETTE_DIR} 안의 팔레트 ({len(files)}개):\n")
    for f in files:                                     # 각 파일
        name = f.replace(".raw", "")                   # 확장자 제거
        # 추천 목록에 있으면 설명 붙이기
        rec = next((r for r in RECOMMENDED if r[0] == name), None)  # 추천 정보
        if rec:
            print(f"  ⭐ {name:14s} {rec[1]} — {rec[2]}")
        else:
            print(f"     {name}")

    print("\n" + "-" * 58)
    print("🎯 목적별 추천:")
    for pal, use, desc in RECOMMENDED:                  # 추천 4종
        print(f"   {use:14s} → {pal}")
    print("\n실행: python test_palette.py --run Iron_Black")
    return True


def run_palette(name):
    """특정 팔레트로 flirone을 실행한다."""
    raw_path = os.path.join(PALETTE_DIR, f"{name}.raw")  # 팔레트 경로
    if not os.path.exists(raw_path):                    # 없으면
        print(f"❌ 팔레트가 없습니다: {name}.raw")
        print("   → python test_palette.py --list 로 목록 확인")
        return False

    # 추천 설명 출력
    rec = next((r for r in RECOMMENDED if r[0] == name), None)  # 추천 정보
    if rec:
        print(f"🎨 {name} — {rec[1]}")
        print(f"   {rec[2]}\n")

    print(f"▶ flirone 실행: {name}.raw")
    print("   (종료하려면 Ctrl+C, 영상 확인은 다른 터미널에서)")
    print(f"   확인: python test_thermal_camera.py\n")
    # flirone은 폴더 안에서 ./ 로 실행해야 함
    cmd = ["sudo", "./flirone", f"./palettes/{name}.raw"]  # 실행 명령
    try:
        subprocess.run(cmd, cwd=FLIR_DIR)              # flirone 폴더에서 실행
    except KeyboardInterrupt:                           # Ctrl+C
        print("\n⏹ flirone 종료")
    except FileNotFoundError:                            # flirone 없음
        print(f"❌ flirone 실행 파일이 없습니다: {FLIR_DIR}/flirone")
        print("   → cd ~/flirone-v4l2 && make 로 빌드하세요")
        return False
    return True


def tour():
    """추천 팔레트 4종을 차례로 보여준다 (비교용)."""
    print("=" * 58)
    print("🎠 팔레트 투어 — 추천 4종을 차례로 실행합니다")
    print("=" * 58)
    print("각 팔레트가 실행되면 다른 터미널에서 영상을 확인하세요:")
    print("  python test_thermal_camera.py\n")
    print("다음 팔레트로 넘어가려면 flirone 창에서 Ctrl+C\n")

    for pal, use, desc in RECOMMENDED:                  # 추천 4종 순회
        print("\n" + "=" * 58)
        print(f"🎨 [{use}] {pal}")
        print(f"   {desc}")
        print("=" * 58)
        input("   ▶ Enter를 누르면 이 팔레트로 실행합니다...")  # 사용자 대기
        run_palette(pal)                               # 실행 (Ctrl+C로 다음)
    print("\n✅ 팔레트 투어 완료! 마음에 드는 팔레트를 골라 쓰세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flirone 팔레트 테스트")  # 파서
    parser.add_argument("--list", action="store_true", help="팔레트 목록 보기")  # 목록
    parser.add_argument("--run", type=str, metavar="NAME", help="특정 팔레트로 실행")  # 실행
    parser.add_argument("--tour", action="store_true", help="추천 4종 순회 비교")  # 투어
    args = parser.parse_args()                          # 파싱

    if args.list:                                       # --list
        sys.exit(0 if list_palettes() else 1)
    elif args.run:                                      # --run NAME
        sys.exit(0 if run_palette(args.run) else 1)
    elif args.tour:                                     # --tour
        tour()
    else:                                               # 옵션 없으면 목록
        list_palettes()
