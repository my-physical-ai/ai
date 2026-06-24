# dent_detector.py
# 생수병 찌그러짐 검출 — 양품 depth 대비 편차를 PyTorch 텐서로 계산하는 룰 기반 로직

import os                       # 파일 경로 처리
import numpy as np              # 수치 배열 처리
import cv2                      # 영상 처리·컬러맵
import torch                    # PyTorch — 텐서 연산 (GPU 가속 가능)
import torch.nn.functional as F # 텐서 풀링·필터 연산


class DentDetector:
    """양품 생수병 depth와 비교해 찌그러짐을 검출하는 클래스."""

    def __init__(self, golden_path=None, depth_scale=0.001,
                 dent_thr_mm=3.0, min_area_px=80, device="cpu"):
        self.depth_scale = depth_scale      # depth 원시값→미터 환산 계수
        self.dent_thr_mm = dent_thr_mm      # 찌그러짐 판정 임계 깊이 (mm)
        self.min_area_px = min_area_px      # 찌그러짐으로 인정할 최소 면적 (픽셀)
        self.device = device                # 연산 장치 ("cpu" 또는 "cuda")
        self.golden_tensor = None           # 양품 기준 depth (torch 텐서)

        # 양품 기준 depth(golden sample) 로드
        if golden_path and os.path.exists(golden_path):
            golden_np = np.load(golden_path)                          # 저장된 양품 depth
            self.golden_tensor = torch.from_numpy(golden_np.astype(np.float32)).to(device)  # 텐서 변환
            print(f"✅ 양품 기준 로드: {golden_path}")
        else:
            print("⚠️ 양품 기준 없음 → register_golden() 먼저 실행 필요")

    def register_golden(self, depth_array, save_path):
        """현재 depth를 양품 기준으로 등록·저장한다 (여러 프레임 평균은 app에서 처리)."""
        self.golden_tensor = torch.from_numpy(depth_array.astype(np.float32)).to(self.device)  # 텐서 보관
        np.save(save_path, depth_array)                              # 파일 저장
        print(f"📌 양품 기준 등록 완료: {save_path}")
        return True

    def detect(self, depth_array):
        """현재 depth에서 찌그러짐을 검출 → 결과 dict 반환."""
        if self.golden_tensor is None:
            return {"ok": False, "msg": "양품 기준이 등록되지 않았습니다"}  # 기준 없으면 중단

        # 현재 depth를 텐서로 변환 (양품과 같은 장치)
        cur = torch.from_numpy(depth_array.astype(np.float32)).to(self.device)
        gold = self.golden_tensor                                    # 양품 기준 텐서

        # --- 1) 유효 영역 마스크 (양쪽 모두 depth > 0 인 픽셀만 비교) ---
        valid = (cur > 0) & (gold > 0)                               # 측정 유효 마스크

        # 유효 픽셀 비율 점검 (투명병·거리 문제 조기 경고)
        # 양품 영역 중 현재 프레임에서 측정된 비율이 너무 낮으면 신뢰 불가
        gold_valid = (gold > 0)                                      # 양품 유효 영역
        if gold_valid.sum() > 0:
            valid_ratio = float(valid.sum()) / float(gold_valid.sum())  # 유효 비율
        else:
            valid_ratio = 0.0
        # 유효 픽셀이 50% 미만이면 측정 신뢰 경고 (투명·반사·거리 문제 의심)
        low_quality = valid_ratio < 0.5

        # --- 2) 부호 있는 편차(mm) 계산: 양품 - 현재 ---
        # 찌그러짐(움푹)이면 표면이 안으로 → 거리(depth)가 멀어짐 → (gold - cur) 음수
        # 튀어나옴이면 표면이 가까워짐 → (gold - cur) 양수
        diff_mm = (gold - cur) * self.depth_scale * 1000             # mm 단위 편차
        diff_mm = torch.where(valid, diff_mm, torch.zeros_like(diff_mm))  # 무효 픽셀은 0

        # --- 3) 절대 편차로 찌그러짐 후보 마스크 (임계값 초과) ---
        abs_dev = torch.abs(diff_mm)                                 # 편차 절대값
        dent_mask = (abs_dev > self.dent_thr_mm) & valid            # 임계 초과 + 유효

        # --- 4) 노이즈 제거: 작은 점들 제거 (최대 풀링으로 침식·팽창 효과) ---
        mask_f = dent_mask.float().unsqueeze(0).unsqueeze(0)        # (1,1,H,W) 형태로
        # 침식(min) → 작은 노이즈 제거: -maxpool(-x) 트릭
        eroded = -F.max_pool2d(-mask_f, kernel_size=3, stride=1, padding=1)
        # 팽창(max) → 남은 영역 복원
        cleaned = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        dent_mask = (cleaned.squeeze() > 0.5)                        # 정제된 마스크

        # --- 5) 결과 집계 (CPU numpy로 변환해 통계) ---
        dent_mask_np = dent_mask.cpu().numpy().astype(np.uint8)      # 마스크 numpy
        abs_dev_np = abs_dev.cpu().numpy()                          # 편차 numpy
        dent_area = int(dent_mask_np.sum())                        # 찌그러짐 면적(픽셀)
        max_dev = float(abs_dev_np[dent_mask_np > 0].max()) if dent_area > 0 else 0.0  # 최대 깊이

        # --- 6) 면적이 너무 작으면 노이즈로 간주 → 정상 ---
        is_dent = dent_area >= self.min_area_px                      # 찌그러짐 여부

        # --- 7) 찌그러짐 위치(외접 사각형) 찾기 ---
        boxes = []
        if is_dent:
            contours, _ = cv2.findContours(dent_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < self.min_area_px:           # 작은 조각 무시
                    continue
                x, y, w, h = cv2.boundingRect(c)                   # 외접 사각형
                # 해당 영역의 최대 깊이 계산
                region = abs_dev_np[y:y+h, x:x+w]
                region_mask = dent_mask_np[y:y+h, x:x+w] > 0
                local_max = float(region[region_mask].max()) if region_mask.any() else 0.0
                boxes.append({"x": x, "y": y, "w": w, "h": h, "depth_mm": round(local_max, 2)})

        return {
            "ok": True,
            "verdict": "BLOCK" if is_dent else "PASS",  # 찌그러짐 있으면 불합격
            "is_dent": is_dent,                         # 찌그러짐 여부
            "max_depth_mm": round(max_dev, 2),          # 최대 찌그러짐 깊이
            "dent_area_px": dent_area,                  # 찌그러짐 면적
            "boxes": boxes,                             # 찌그러짐 위치들
            "diff_map": diff_mm.cpu().numpy(),          # 편차 맵 (히트맵용)
            "valid_ratio": round(valid_ratio, 2),       # 유효 픽셀 비율 (측정 품질)
            "low_quality": low_quality,                 # 품질 경고 (투명·거리 문제)
        }

    def make_heatmap(self, diff_map):
        """편차 맵을 컬러 히트맵으로 변환한다 (시각화용)."""
        abs_map = np.abs(diff_map)                                  # 절대 편차
        # 0~임계값*2 범위로 정규화 (임계값 부근이 잘 보이게)
        norm = np.clip(abs_map / (self.dent_thr_mm * 2), 0, 1) * 255
        norm = norm.astype(np.uint8)                               # uint8 변환
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)        # JET 컬러맵
        return heatmap                                             # 파랑=정상, 빨강=찌그러짐
