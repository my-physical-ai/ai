# LeKiwi 비전 파이프라인 — YOLO 추적 + GDINO 언어탐지 + SAM3 정밀분할

3대 머신이 역할을 나눠 동작하는 실시간 비전 시스템입니다.
UI에서 **2단계로 물체를 찾은 뒤, "SAM3 정밀분할" 버튼**을 누르면
4090의 SAM3가 픽셀 단위 경계를 만들어 돌려줍니다.

```
┌─────────────┐   ZeroMQ    ┌──────────────────┐   HTTP POST   ┌─────────────────┐
│   Pi 5      │  카메라영상   │   NUC (Intel)    │   박스 전송    │   RTX 4090      │
│ 카메라 캡처  │ ──5556/5557→ │ YOLO+추적+GDINO  │ ──5001/segment→│   SAM3 분할     │
│             │             │  Flask :5000     │ ←마스크 회신──  │  Flask :5001    │
└─────────────┘             └──────────────────┘               └─────────────────┘
                                     │
                              브라우저 http://NUC_IP:5000
```

## 디렉토리 구조

```
lekiwi_sam3/
├── nuc/                      ← NUC(Intel CPU)에서 실행
│   ├── app.py                ← Flask 메인 서버
│   ├── botsort_lekiwi.yaml    ← BoT-SORT 설정 (있으면 자동생성 안 함)
│   └── templates/
│       └── index.html        ← 3단계 파이프라인 UI (SAM3 버튼 포함)
├── 4090/                     ← RTX 4090(GPU)에서 실행
│   └── sam3_server.py        ← SAM3 정밀분할 서버
├── pi/                       ← Raspberry Pi 5에서 실행
│   ├── send_camera_front.py  ← FRONT 카메라 (/dev/video0 → 5556)
│   └── send_camera_top.py    ← TOP 카메라 (/dev/video3 → 5557)
└── README.md
```

---

## 실행 순서

### ① Pi 5 — 카메라 전송 (터미널 2개)

```bash
conda activate lerobot

# 터미널 1: FRONT 카메라
python send_camera_front.py

# 터미널 2: TOP 카메라
python send_camera_top.py
```

### ② RTX 4090 — SAM3 서버

```bash
conda activate sam3            # SAM3 설치된 환경
pip install flask              # 아직 없으면

python sam3_server.py
# → http://0.0.0.0:5001 에서 대기
```

> ⚠️ **sam3_server.py의 `load_sam3_model()`** 은 설치 방식(Ultralytics / 공식 sam3 패키지)에
> 따라 자동 폴백하도록 작성돼 있습니다. 공식 패키지의 박스 프롬프트 API명만
> 본인 버전에 맞게 한 번 확인하세요.

### ③ NUC — 메인 서버

```bash
conda activate lerobot
pip install requests           # 아직 없으면

python app.py
# → http://0.0.0.0:5000
```

### ④ 브라우저

```
http://<NUC IP>:5000
```

---

## ★ 실행 전 반드시 수정할 값

| 파일 | 변수 | 설명 |
|------|------|------|
| `nuc/app.py` | `PI_IP` | Pi 5 실제 IP (`hostname -I`) |
| `nuc/app.py` | `SAM3_SERVER_URL` | **4090 PC 실제 IP** (예: `http://192.168.50.200:5001`) |
| `nuc/app.py` | `YOLO_MODEL_PATH` | YOLO26 OpenVINO 모델 절대경로 |
| `4090/sam3_server.py` | `SAM3_CHECKPOINT` | SAM3 체크포인트 경로 |
| `pi/send_camera_top.py` | `--device` | TOP 카메라 장치 (기본 `/dev/video3`) |

---

## 사용 흐름 (UI) — 박스 vs 마스크 정밀분할 비교

이 UI는 **"박스(GDINO)와 픽셀 마스크(SAM3)의 차이를 눈으로 증명"** 하는 데 집중합니다.

1. **물체 올리기** — 카메라 앞에 기판(PCB) 같은 물체를 비스듬히 놓습니다.
   (비스듬할수록 박스가 배경을 많이 포함해 효과가 큽니다.)
2. **박스로 위치 찾기** — 영어로 물체 이름 입력(`circuit board`, `green pcb` 등) → GDINO가 박스 탐지.
3. **정밀분할 실행** — 버튼 클릭 → 4090 SAM3가 픽셀 마스크 생성 → 결과 표시:
   - **핵심 수치**: "박스의 N%가 배경이었습니다"
   - **좌우 비교**: 왼쪽 빨강 박스(배경 포함) / 오른쪽 초록 마스크(물체만)
   - **정확도 바**: 박스 100% 대비 마스크가 차지하는 실제 물체 비율
   - **물체별 표**: 박스 면적 vs 마스크 면적 vs 물체 비율

> **교육 포인트**: 박스는 사각형이라 비스듬한 물체를 감싸면 모서리에 배경이 잔뜩 들어갑니다.
> SAM3 마스크는 물체 실루엣만 픽셀 단위로 따내므로, "박스로는 30% 배경, 마스크로는 0% 배경"
> 같은 차이를 숫자로 체감할 수 있습니다. 로봇이 물체를 집을 때 이 차이가 정확도를 좌우합니다.

> SAM3 버튼은 **4090 서버가 온라인 + 박스 탐지 성공** 시에만 활성화됩니다.

---

## 동작 안 할 때 체크리스트

| 증상 | 확인 |
|------|------|
| SAM3 버튼이 계속 비활성 | 상태바 `SAM3 (4090)` 점 색 → 빨강이면 4090 서버/IP 확인 |
| "SAM3 서버 응답 없음" | `app.py`의 `SAM3_SERVER_URL` IP가 4090 실제 IP인지 |
| 카메라 점 빨강 | Pi에서 send_camera 실행됐는지, `PI_IP` 맞는지 |
| 영상 안 나옴 | 방화벽에서 5556/5557 포트 열렸는지 |
| SAM3 너무 느림 | 4090이 유선 연결인지, 첫 호출은 모델 워밍업으로 느릴 수 있음 |

---

## 핵심 설계 메모

- **fallback 안전장치**: 4090이 꺼져 있어도 NUC 단독으로 추적·GDINO는 정상 동작.
  SAM3 버튼만 비활성화됨 (교육 현장에서 4090 없이도 수업 가능).
- **이미지 임베딩 1회**: SAM3는 `set_image`가 무겁고 박스별 `predict`는 가벼움 →
  박스 여러 개여도 임베딩은 1번만 (sam3_server.py 루프 구조).
- **마지막 박스 재사용**: GDINO 탐지 시 NUC가 프레임+박스를 기억 → SAM3 버튼 클릭 시
  이미지 재전송 없이 그 결과를 4090으로 보냄.
- **유선 권장**: NUC↔4090 사이 base64 이미지가 오가므로 4090도 AX1800에 유선 연결.
