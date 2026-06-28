# ============================================================
# 파일명: sam3_client.py
# 설명: 4090 SAM3 서버(192.168.0.75:5001)에 박스를 보내 사람 윤곽 마스크를 받는다
# 원리: YOLO 박스 → SAM3 add_geometric_prompt → 픽셀 윤곽 → 배경 제외 측정
# 서버: 192.168.0.75:5001 — form-data(image,box) 전송, packbits 압축 마스크 회신
# 변경 이력:
#   - [2026-06-23] 실제 sam3_server.py 형식에 맞춤 (form-data + packbits)
# ============================================================

import requests                                          # HTTP 호출
import cv2                                               # 영상 인코딩
import numpy as np                                       # 배열/압축 해제

SAM3_SERVER = "http://192.168.0.75:5001"                 # 4090 SAM3 서버 (실제 IP)
SAM3_TIMEOUT = 30                                        # 첫 호출 워밍업 대비 30초


def check_sam3_online():
    """SAM3 서버가 살아있는지 확인한다 (UI 버튼 활성화 판단용)."""
    try:
        r = requests.get(f"{SAM3_SERVER}/ping", timeout=3)  # 헬스체크 (/ping)
        return r.status_code == 200                      # 200이면 온라인
    except Exception:                                    # 연결 실패
        return False                                     # 오프라인


def request_one_mask(vis_bgr, box, percentile=92):
    """박스 1개를 SAM3로 보내 사람 윤곽 마스크 1개를 받는다 (form-data 방식)."""
    # 영상을 JPEG 바이트로 인코딩 (서버가 request.files["image"]로 받음)
    ok, jpeg = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])  # JPEG 인코딩
    if not ok:                                           # 인코딩 실패
        return None
    files = {"image": ("frame.jpg", jpeg.tobytes(), "image/jpeg")}  # 파일 파트
    # 박스를 "x1,y1,x2,y2" 문자열로 (서버가 request.form["box"]로 받음)
    data = {"box": f"{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}",  # 박스 파트
            "percentile": str(percentile)}              # 백분위수 임계값 (상위 8%=인물)
    try:
        r = requests.post(f"{SAM3_SERVER}/segment",      # SAM3 분할 요청
                          files=files, data=data, timeout=SAM3_TIMEOUT)
        r.raise_for_status()                            # HTTP 에러 체크
        resp = r.json()                                 # 응답 파싱
        area = resp.get("area_ratio")                   # 분할 면적 비율
        if area is not None:                            # 면적 정보 있으면
            print(f"  🎯 SAM3 윤곽 면적 {area*100:.1f}%")  # 품질 확인 로그
        return decode_packed_mask(resp)                 # packbits 마스크 복원
    except requests.exceptions.ConnectionError:         # 연결 실패
        print("⚠️ SAM3 서버 연결 실패 (192.168.0.75 확인) → 박스 측정 폴백")
        return None
    except requests.exceptions.Timeout:                 # 시간 초과
        print(f"⚠️ SAM3 응답 시간 초과 ({SAM3_TIMEOUT}초) → 박스 측정 폴백")
        return None
    except Exception as e:                              # 기타 오류
        print(f"⚠️ SAM3 오류: {e} → 박스 측정 폴백")
        return None


def decode_packed_mask(resp):
    """서버의 packbits 압축 마스크를 numpy bool 배열로 복원한다."""
    masks = resp.get("masks", [])                       # 압축 마스크 리스트
    if not masks:                                        # 마스크 없으면 (분할 실패)
        return None
    w = resp.get("width")                               # 원본 너비
    h = resp.get("height")                              # 원본 높이
    if w is None or h is None:                           # 크기 정보 없으면
        return None
    # 첫 마스크 복원: packbits → unpackbits → h*w로 자르고 reshape
    packed = np.array(masks[0], dtype=np.uint8)         # 압축 바이트 배열
    flat = np.unpackbits(packed)[:h * w]                # 비트 풀기 + 픽셀 수만큼
    return flat.reshape(h, w).astype(bool)              # (H,W) bool 마스크 반환


def request_person_masks(vis_bgr, boxes, percentile=92):
    """여러 박스를 SAM3로 보내 사람 윤곽 마스크 리스트를 받는다 (박스마다 1회 호출)."""
    if not boxes:                                        # 박스 없으면
        return None
    masks = []                                           # 마스크 리스트
    for box in boxes:                                    # 각 박스마다
        m = request_one_mask(vis_bgr, box, percentile)  # SAM3 마스크 1개 요청 (백분위수 전달)
        if m is None:                                    # 하나라도 실패하면
            return None                                  # 전체 폴백 (박스 측정으로)
        masks.append(m)                                  # 마스크 누적
    return masks                                         # 마스크 리스트 반환


def request_person_mask_text(vis_bgr, prompt="person"):
    """SAM3 텍스트 프롬프트로 사람 윤곽 1개를 받는다 (박스 없이 의미 기반).

    박스 프롬프트가 사람 열화상에서 윤곽을 못 따는 문제 해결:
    "person" 텍스트로 SAM3가 사람을 의미적으로 찾아 정확히 분할.

    Args:
        vis_bgr: 열화상 영상 (BGR)
        prompt: 분할 개념 (기본 "person")

    Returns:
        mask: 사람 윤곽 (HxW bool) 또는 None (폴백 신호)
    """
    ok, jpeg = cv2.imencode(".jpg", vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])  # JPEG 인코딩
    if not ok:                                           # 인코딩 실패
        return None
    files = {"image": ("frame.jpg", jpeg.tobytes(), "image/jpeg")}  # 파일 파트
    data = {"prompt": prompt}                            # 텍스트 프롬프트 (사람)
    try:
        r = requests.post(f"{SAM3_SERVER}/segment_text",  # 텍스트 분할 요청
                          files=files, data=data, timeout=SAM3_TIMEOUT)
        r.raise_for_status()                            # HTTP 에러 체크
        resp = r.json()                                 # 응답 파싱
        area = resp.get("area_ratio")                   # 분할 면적
        if area is not None:                            # 면적 정보 있으면
            print(f"  🗣️ SAM3 텍스트('{prompt}') 윤곽 면적 {area*100:.1f}%")  # 품질 로그
        return decode_packed_mask(resp)                 # packbits 마스크 복원
    except requests.exceptions.ConnectionError:         # 연결 실패
        print("⚠️ SAM3 서버 연결 실패 (192.168.0.75) → 박스 측정 폴백")
        return None
    except requests.exceptions.Timeout:                 # 시간 초과
        print(f"⚠️ SAM3 응답 시간 초과 → 박스 측정 폴백")
        return None
    except Exception as e:                              # 기타 오류
        print(f"⚠️ SAM3 텍스트 분할 오류: {e} → 박스 측정 폴백")
        return None

