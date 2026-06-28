# ============================================================
# 파일명: vlm_judge.py
# 설명: IRGPT 대체 — 192.168.0.36 InternVL 서버에 열화상 판정을 맡기는 클라이언트
# 원리: 절대온도 없이 열화상 패턴(색분포·대칭·국소과열)으로 VLM이 판단
# 서버: 192.168.0.36:5002 (InternVL2.5, 규칙기반 폴백 내장)
# 변경 이력:
#   - [2026-06-23] 서버 IP 0.74→0.36 수정, 응답 형식 analysis/engine에 맞춤
# ============================================================

import base64                                           # 이미지 인코딩
import requests                                          # HTTP 호출

import cv2                                               # 영상 처리
import numpy as np                                       # 배열 연산

VLM_SERVER = "http://192.168.0.36:5002/analyze"          # ← InternVL 판정 서버 (실제 IP)

# 열화상 판정 프롬프트 (공장 종합 안전 관리자 역할 — 차분한 전문가 톤)
# VLM-IRIS(arXiv 2025.12) 연구: 역할 부여 + 프롬프트 정교화가 판정 정확도를 좌우함
THERMAL_JUDGE_PROMPT = """당신은 제조 공장의 종합 안전을 책임지는 베테랑 안전관리자입니다.
작업자의 건강 상태와 현장 설비의 열 이상을 함께 살피며, 매일 현장을 순회하면서
열화상으로 위험 신호를 미리 포착해 사고를 예방하는 것이 당신의 역할입니다.

지금 보고 있는 것은 작업자의 열화상(thermal) 이미지입니다. 색이 밝을수록(노랑·흰색)
뜨겁고 어두울수록(보라·검정) 차갑습니다. 절대 온도(℃)는 측정되지 않으므로,
오랜 경험으로 부위 간 상대적인 열 패턴을 읽어 판단하십시오.

관리자로서 다음을 종합적으로 살피되, 체크리스트를 나열하지 말고
현장 경험에서 우러난 자연스러운 판단으로 서술하십시오.
- 몸의 좌우 열 균형이 맞는가, 한쪽만 치우쳐 뜨겁거나 차가운가
- 주변보다 유독 밝게 타오르는 국소 과열 부위가 있는가 (염증·마찰·과부하 신호)
- 상체는 따뜻하고 말단은 덜 따뜻한 정상 분포에서 벗어났는가
- 작업자의 전반적 열 상태가 안전하게 작업을 이어갈 수준인가

[답변 형식 — 차분하고 전문적인 안전 리포트 톤]
1. 첫 문장: '정상' 또는 '주의' 또는 '경고' 중 하나로 종합 판정
2. 그 판단의 근거를 현장 관리자의 시선으로 2~3문장 서술
3. 필요 시 후속 조치나 관찰 권고를 1문장
전체 4문장 이내, 한국어로, 침착하고 신뢰감 있는 어조로 작성하십시오."""


def img_to_b64(img_bgr):
    """BGR 이미지를 base64 JPEG 문자열로 변환한다 (서버 전송용)."""
    _, jpeg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])  # JPEG 인코딩
    return base64.b64encode(jpeg.tobytes()).decode()    # base64 문자열 반환


def judge_with_vlm(thermal_img, zones_info=None, timeout=30):
    """192.168.0.36 InternVL 서버에 열화상 이미지를 보내 판정 결과를 받는다."""
    payload = {                                          # 서버 전송 데이터
        "image": img_to_b64(thermal_img),               # 열화상 이미지
        "prompt": THERMAL_JUDGE_PROMPT,                 # 판정 프롬프트
    }
    if zones_info:                                       # 3등분 측정값 있으면 힌트로 첨부
        # 측정된 상대지수를 점검 관점과 연결해 구조화 (VLM 판단 보조)
        parts = [f"{z['name']}={z['mean']}" for z in zones_info
                 if z.get("mean") is not None]
        hint = "측정된 부위별 상대지수(0~100): " + ", ".join(parts)  # 수치 힌트
        # 부위 간 편차를 계산해 국소 과열 판단 보조
        vals = [z["mean"] for z in zones_info if z.get("mean") is not None]
        if len(vals) >= 2:                              # 2구역 이상이면
            spread = max(vals) - min(vals)              # 최대-최소 편차
            hint += f". 부위 간 최대 편차는 {spread}입니다"  # 편차 정보 추가
            if spread >= 40:                            # 편차 크면 경고 힌트
                hint += " (편차가 커서 국소 과열 가능성 검토 필요)"
        payload["hint"] = hint                          # 구조화된 힌트 전달
    try:
        r = requests.post(VLM_SERVER, json=payload, timeout=timeout)  # 서버 호출
        r.raise_for_status()                            # HTTP 에러 체크
        return {"ok": True, "result": r.json()}         # 판정 결과 반환
    except requests.exceptions.ConnectionError:         # 서버 연결 실패
        return {"ok": False, "error": "VLM 서버 연결 실패 (192.168.0.36 서버 실행 확인)"}
    except requests.exceptions.Timeout:                 # 응답 시간 초과
        return {"ok": False, "error": f"VLM 응답 시간 초과 ({timeout}초)"}
    except Exception as e:                              # 기타 오류
        return {"ok": False, "error": f"VLM 판정 오류: {e}"}


def format_verdict(vlm_result):
    """VLM 판정 결과를 화면 표시용 한국어 텍스트로 변환한다."""
    if not vlm_result.get("ok"):                        # 실패 시
        return f"⚠️ VLM 판정 불가: {vlm_result.get('error', '알 수 없음')}"
    res = vlm_result["result"]                          # 판정 내용
    # 서버 응답 형식: {"analysis": "판정텍스트", "engine": "internvl" 또는 "rule_based"}
    analysis = res.get("analysis", "")                  # 판정 텍스트
    engine = res.get("engine", "?")                     # 사용 엔진
    tag = "🤖 VLM" if engine == "internvl" else "📐 규칙기반"  # 엔진 표시
    return f"[{tag}]\n{analysis}"                        # 엔진 + 판정 조합
