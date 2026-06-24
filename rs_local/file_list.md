# 📋 RealSense LOCAL (Type A) — 파일 리스트
# [2026-02-21 생성] NUC 직결 미션 1~7 테스트 패키지

---

## 📁 폴더 구조

```
rs_local/
├── file_list.md          ← 이 파일 (전체 파일 목록 + 설명)
├── app.py                ← Flask 메인 서버 (미션 1~7 전체)
├── test_realsense.py     ← RealSense 사전 테스트 스크립트
├── requirements.txt      ← pip 의존성 목록
└── templates/
    └── index.html        ← 웹 UI (미션 카드 + MJPEG 스트림)
```

---

## 📄 파일별 설명

| # | 파일명 | 줄 수 | 역할 |
|---|--------|-------|------|
| 1 | `app.py` | ~640 | Flask 메인 서버. RealSense 초기화, 미션 1~7 프레임 처리, MJPEG 스트리밍, 클릭/키/ROI/슬라이더 API |
| 2 | `templates/index.html` | ~240 | 웹 UI. 미션 카드 7개, MJPEG 스트림 표시, 클릭 좌표 전송, 컬러맵 버튼, ROI 드래그, 근접 경고 슬라이더 |
| 3 | `test_realsense.py` | ~130 | 사전 테스트. 6가지 항목(import, 연결, 파이프라인, 스케일, 프레임, 필터)을 PASS/FAIL로 검증 |
| 4 | `requirements.txt` | 5 | pip 의존성. pyrealsense2, flask, opencv-python, numpy |
| 5 | `file_list.md` | - | 이 파일. 전체 파일 목록과 실행 순서 안내 |

---

## 🚀 실행 순서

```
1. conda activate lerobot              ← 가상환경 활성화
2. pip install -r requirements.txt      ← 의존성 설치
3. python3 test_realsense.py            ← 사전 테스트 (6항목 PASS 확인)
4. python3 app.py                       ← Flask 서버 실행
5. 브라우저에서 http://localhost:5000    ← 미션 카드 클릭으로 시작
```

---

## 🎯 미션 구성 (7개)

| 미션 | 이름 | 인터랙션 | 출력 이미지 |
|------|------|----------|------------|
| 1 | RGB + Depth 듀얼 뷰 | 없음 | 1280×480 (좌우 결합) |
| 2 | 클릭 거리 측정 | 클릭 | 640×480 (단일) |
| 3 | 2D→3D 좌표 변환 | 클릭 | 640×480 (단일) |
| 4 | Depth 컬러맵 변경 | 키보드 1~4 | 1280×480 (좌우 결합) |
| 5 | ROI 영역 통계 | 드래그 | 640×480 (단일) |
| 6 | 필터 비교 | 없음 | 1280×480 (좌우 결합) |
| 7 | 근접 경고 | 슬라이더 | 640×480 (단일) |

---

## 🔧 수정이 필요한 설정값 (app.py 상단)

```python
FLASK_HOST = "0.0.0.0"   # ← NUC만 접속: "127.0.0.1"
FLASK_PORT = 5000         # ← 포트 충돌 시 변경
CAMERA_W = 640            # ← 해상도 변경 가능 (320~1280)
CAMERA_H = 480            # ← 해상도 변경 가능 (240~720)
CAMERA_FPS = 30           # ← 프레임 레이트 변경 가능 (15/30)
```

---

## ⚙️ 하드웨어 요구사항

- **NUC**: Intel NUC10i7FNH (또는 i5 이상 Intel CPU)
- **카메라**: Intel RealSense D435i / D455 / D405
- **USB**: USB 3.0 포트 (파란색) 필수
- **OS**: Ubuntu 20.04 / 22.04 / 24.04
- **Python**: 3.8 이상 (conda lerobot 환경 권장)
