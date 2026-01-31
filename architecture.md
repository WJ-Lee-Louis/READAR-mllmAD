# READ + MVTecAD Anomaly Detection Pipeline (Architecture)

이 문서는 READ 모델을 MVTecAD Anomaly Detection task에 적용한 전체 과정을 **모델 아키텍처 관점**에서 정리한 것이다. 
(코드 기준: `READ-main` 폴더)

---

## 1. 전체 파이프라인 개요

**입력 이미지(MVTecAD test)** → **CLIP 비전 인코더(vision tower)** → **LLaVA 기반 멀티모달 LLM** → **SasP/Similarity 기반 포인트 생성** → **SAM 프롬프트(포인트+<SEG> 토큰 임베딩)** → **SAM 마스크 디코더** → **logit 저장(.npy)** → **벤치마크 지표 계산**

핵심은 **CLIP 기반 시각 피처**와 **LLM 텍스트 피처**가 결합되어 **SAM 디코더의 프롬프트 입력(포인트 + 텍스트 임베딩)**을 형성하고, 최종 **세그멘테이션 logits**가 저장/평가된다는 점이다.

---

## 2. 입력 데이터 및 MVTecAD 구조

- MVTecAD 테스트셋 구조:
  - `MVTecAD/<class>/test/<anomaly_type>/*.png`
  - `MVTecAD/<class>/test/good/*.png`
  - 이상 유형(anomaly)은 `ground_truth`에 대응 GT 마스크 존재
  - good은 GT 마스크 없음

---

## 3. 비전 타워 (CLIP) 처리 흐름

### 3.1 비전타워 설정
- 사용 모델: `clip-vit-large-patch14-336`
- 설정 파일(`clip-vit-large-patch14-336/config.json`):
  - `image_size = 336`
  - `patch_size = 14`
  - 패치 수: `(336 / 14)^2 = 24 x 24 = 576` (CLS 포함 시 577 토큰)

### 3.2 CLIP 전처리
- `ImageProcessor.load_and_preprocess_image()` 내부:
  - `CLIPImageProcessor.preprocess()` 사용
  - resize 및 center crop을 통해 **336x336**으로 정규화

---

## 4. SAM 처리 흐름

- SAM 입력 사이즈는 **1024x1024** 고정
- `ResizeLongestSide(1024)` → padding → 1024x1024 입력
- SAM 내부 `patch_size = 16`
- SAM encoder의 패치 그리드: **64x64**

---

## 5. LLaVA/READ 텍스트-비전 결합

### 5.1 입력 구조
- 프롬프트 예시: `Segment the {anomaly_type} on the {class_name}.`
- 이미지 토큰 + 텍스트 토큰으로 구성
- LLaVA가 생성한 **<SEG> 토큰 임베딩**이 SAM 디코더에 전달됨

### 5.2 Visual Embedding + Text Embedding
- CLIP vision tower 출력 feature와 <SEG> 토큰 임베딩을 결합하여 SAM 프롬프트 조건을 구성

---

## 6. SasP / Similarity 기반 포인트 생성

- CLIP 기반 similarity map을 생성하고
- 높은 activation 영역을 기반으로 포인트 후보 선정
- `Discrete_to_Continuous` 과정을 거쳐 **연속 좌표값(point)** 으로 보정
- 최종 포인트는 SAM prompt encoder에 입력됨

---

## 7. SAM 디코더 입력 구조

### 입력 프롬프트
- **포인트 좌표 + 라벨(point_coords, point_labels)**
- **<SEG> 토큰 임베딩**

### 처리 흐름
- `PromptEncoder` → `MaskDecoder`
- `postprocess_masks()`를 통해 원본 이미지 크기로 복원

---

## 8. 출력과 저장

### 8.1 출력 형식
- **raw logits** (`pred_mask`)이 `.npy`로 저장됨
- 저장 위치: `read_mvtec_outputs_*` 폴더

### 8.2 로그 구조
- 각 샘플 별:
  - `metadata.json`
  - `pred_mask.npy` (raw logit)
  - 시각화 이미지

---

## 9. 벤치마크 지표 계산

### 사용 지표
- iAUROC (image-level)
- pAUROC (pixel-level)
- PRO (region overlap)
- F1-max
- (옵션) PRO_hard

### 특징
- **pAUROC / iAUROC**: 픽셀/이미지 score 순서만 중요
- **PRO / F1-max**: threshold sweep에 민감
- PRO는 **전체 테스트셋 기준**으로 region-level 계산

---

## 10. 요약

- **CLIP**: 336x336 입력, patch 14 → 24x24 (576 patch)
- **SAM**: 1024x1024 입력, patch 16 → 64x64
- **LLM**: 텍스트 프롬프트 + <SEG> 임베딩 생성
- **SasP**: similarity 기반 포인트 생성 후 SAM 프롬프트로 전달
- **Output**: raw logits 저장 후 MVTecAD 벤치마크 지표 계산

---

## 참고 코드 위치
- `dataloaders/base_dataset.py`: CLIP/SAM 전처리
- `model/llava/model/multimodal_encoder/clip_encoder.py`: CLIP patch 수 정의
- `model/segment_anything/build_sam.py`: SAM input size 및 patch_size
- `model/READ.py`: <SEG> 토큰 처리 및 SAM 디코더 연결
- `eval/eval_metrics.py`: 지표 계산
