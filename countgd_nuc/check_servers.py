# 컵 재고조사 시스템 — 4090/VLM 서버 연결을 한 번에 점검하는 진단 스크립트
# [2026-06-28 작성] 5단계 파이프라인이 의존하는 모든 AI 서버의 상태를 확인한다

import sys
import requests   # HTTP 요청으로 각 서버의 health를 확인

# ============================================================
# ★ 점검할 서버 목록 (본인 환경에 맞게 IP 수정)
# ============================================================
SERVERS = [
    # (이름, URL, 어느 단계에서 쓰는지)
    ("GDINO",   "http://192.168.0.75:5004/ping",   "2단계 — 컵 재확인"),
    ("SAM3",    "http://192.168.0.75:5001/health", "3단계 — 컵 분할"),
    ("CountGD", "http://192.168.0.75:5005/health", "4단계 — 객체탐지 카운팅"),
    ("VLM",     "http://192.168.0.36:5002/health", "4단계 — 맥락추론 카운팅"),
]


def check_one(name, url, stage, timeout=5):
    """서버 1개의 연결 상태를 확인하고 결과를 출력한다."""
    try:
        r = requests.get(url, timeout=timeout)          # health 요청
        if r.status_code == 200:
            print(f"  ✅ {name:9s} 정상  ({stage})")
            return True
        print(f"  ⚠️ {name:9s} 응답이상 (HTTP {r.status_code}) — {stage}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  ❌ {name:9s} 연결거부 — 서버가 꺼져 있습니다 ({stage})")
        return False
    except Exception as e:
        print(f"  ❌ {name:9s} 오류: {e} ({stage})")
        return False


def main():
    """모든 서버를 순서대로 점검하고 종합 결과를 보여준다."""
    print("=" * 56)
    print("🔍 컵 재고조사 — AI 서버 연결 점검")
    print("=" * 56)
    ok_count = 0
    for name, url, stage in SERVERS:
        if check_one(name, url, stage):
            ok_count += 1
    print("-" * 56)
    print(f"결과: {ok_count}/{len(SERVERS)} 서버 정상")
    if ok_count == len(SERVERS):
        print("🎉 모든 서버 준비 완료! app_step.py를 실행하세요.")
    else:
        print("⚠️ 꺼진 서버를 켠 뒤 다시 실행하세요. (4090/VLM PC 확인)")
    print("=" * 56)
    return 0 if ok_count == len(SERVERS) else 1


if __name__ == "__main__":
    sys.exit(main())   # 모든 서버 정상이면 0, 아니면 1로 종료
