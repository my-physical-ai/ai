# 열화상 3등분 온도측정 AI — NUC 최종본 (v2026-06-23)

YOLO 실시간 탐지 + 상대지수 재정규화 + InternVL 이미지 판정 파이프라인.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `app.py` | 메인 Flask 서버 (YOLO 실시간 + 판별) |
| `thermal_level.py` | inferno RGB → 상대지수 역변환 (온도 순위 보존) |
| `zone_measure.py` | 박스 내 재정규화 3등분 측정 (압축된 지수 펴기) |
| `vlm_judge.py` | InternVL 서버(192.168.0.36) 판정 호출 |
| `templates/index.html` | 웹 UI (실시간 영상 + 판정 표시) |

## 적용 방법

```bash
# 1. NUC 파이프라인 폴더로 이동
cd ~/thermal_3zone_ai/02_nuc_pipeline

# 2. 기존 파일 백업 (혹시 모르니)
cp app.py app.py.bak_$(date +%Y%m%d)

# 3. 이 폴더의 파일을 모두 복사 (templates 포함)
cp app.py thermal_level.py zone_measure.py vlm_judge.py ~/thermal_3zone_ai/02_nuc_pipeline/
cp templates/index.html ~/thermal_3zone_ai/02_nuc_pipeline/templates/

# 4. config.yaml은 기존 것 그대로 사용 (수정 불필요)
```

## 실행 전 확인

```bash
# InternVL 서버(192.168.0.36)가 떠 있는지 확인
curl http://192.168.0.36:5002/health
# → {"status": "ready", "model": "OpenGVLab/InternVL2_5-4B"} 면 정상
```

## 실행

```bash
conda activate lerobot   # 가상환경
flirone                  # FLIR 드라이버 (별도 터미널)
python app.py            # 메인 서버
# 브라우저: http://192.168.0.65:5000
```

## 동작 흐름

```
사람 탐지(YOLO) → 박스 3등분 재정규화 지수(0~100) 표시
              → 열화상 이미지를 192.168.0.36 서버로 전송
              → InternVL이 이미지 보고 판정 → 화면 표시
```

## 핵심 변경점 (이전 버전 대비)

1. BGR2GRAY 제거 → inferno 역변환으로 온도 순위 역전 해결
2. 전역 0~100 환산 → 박스 내 재정규화로 압축된 지수 펴기
3. IRGPT 규칙기반(℃ 가정) → InternVL 이미지 판정으로 교체

## 의존성

```bash
pip install requests pyyaml --break-system-packages
# cv2, numpy, ultralytics, flask는 기존 lerobot 환경에 이미 설치됨
```
