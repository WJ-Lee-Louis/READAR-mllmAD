# READAR: Can Multimodal Large Language Models Reason to Attend to Anomaly Regions?

This repository contains the minimal code needed to run inference with MVTecAD Dataset via READ(Reasoning to Attend) and compute benchmark metrics
(iAUROC, pAUROC, PRO, F1-max). Large artifacts such as datasets, model weights, and inference outputs are intentionally
excluded.

## 1. Background & Motivation
<img width="913" height="387" alt="image" src="https://github.com/user-attachments/assets/987ceb31-9434-47f8-9ad2-80be8477fd31" />

### Unsupervised Anomaly Detection

In real-world industrial and manufacturing environments, collecting sufficient anomaly samples is inherently difficult due to data scarcity and high acquisition costs. As a result, many practical anomaly detection settings assume that anomaly samples are unavailable during training, while test sets contain a mixture of normal and anomalous instances.

This constraint has led to extensive research on **unsupervised anomaly detection**, where models are trained exclusively on normal data and are expected to identify deviations at inference time. Such approaches have demonstrated robust and reliable performance across various industrial benchmarks.

---

### Taxonomy of Unsupervised Anomaly Detection Methods

Unsupervised anomaly detection methods can be broadly categorized into two main paradigms:

#### 1) Feature-Embedding-Based Methods

Feature-embedding-based approaches model the distribution of normal data in a learned feature space and detect anomalies based on deviations from this distribution. Representative techniques include: teacher-student architecture, one-class classification, distribution modeling(e.g., normalizaing flow-based), and feature memory bank-based. Among these, **feature memory bank approaches** such as **SPADE**, **PaDiM**, and **PatchCore** have achieved particularly strong **full-shot** performance. In some benchmarks, these methods demonstrate near-saturated performance, indicating the maturity and effectiveness of feature-based unsupervised anomaly detection.

#### 2) Reconstruction-Based Methods

Reconstruction-based methods learn to reconstruct normal images and use reconstruction errors as anomaly scores. This category includes approaches based on: autoencoder, GAN, transformer, diffusion. Despite their conceptual simplicity, reconstruction-based methods often face challenges related to reconstruction fidelity, training stability, and precise anomaly localization, especially for high-resolution industrial imagery.

---

### Vision–Language Models for Anomaly Detection

Recently, **Vision–Language Models (VLMs)** such as CLIP have been introduced into anomaly detection research. These methods leverage the alignment between **textual descriptions of normal/anomalous states** and **image representations** to enable **few-shot** or **zero-shot anomaly detection**: **WinCLIP**, **AnomalyCLIP**, etc. These approaches significantly improve data efficiency by exploiting pre-trained vision–language alignment. However, they primarily frame anomaly detection as a similarity-based scoring problem, where reasoning and semantic understanding play only a limited role.

---

### Multimodal LLMs for Anomaly Detection

More recent efforts aim to apply **multimodal Large Language Models (LLMs)** directly to anomaly detection tasks. A notable example is **AnomalyGPT**, which explores the integration of multimodal LLMs into anomaly detection pipelines.

The motivation for adopting multimodal LLMs in anomaly detection can be summarized as follows:
> 1) **Generalization and Adaptability**  
   The ability to generalize to unseen products, defect types, and domains without retraining.
> 2) **Explainability and Interaction**  
   Providing natural language explanations and enabling question–answering capabilities regarding detected anomalies.

While AnomalyGPT demonstrates progress toward overcoming the one-class-one-model paradigm and enables general anomaly scoring and localization, the LLM does not yet function as the primary reasoning agent in the anomaly detection process.

---

### Project Objective

This project aims to move beyond CLIP-based anomaly detection frameworks and investigate whether **multimodal foundation models can serve as the core reasoning component** in anomaly detection tasks.

In particular, we focus on:
- **Anomaly segmentation and localization**
- **Training-free and zero-shot settings**

---

### Model Selection

To this end, we adopt a multimodal foundation model pipeline consisting of:
- **LLaVA** as the backbone LLM, leveraging visual instruction tuning
- **SAM (Segment Anything Model)** for precise segmentation and localization

We evaluate several models that integrate these components:
- **LISA**
<img width="1671" height="619" alt="image" src="https://github.com/user-attachments/assets/c2b86a25-18de-4e86-9cb2-720763370cac" />

- **PixelLM**
<img width="1585" height="687" alt="image" src="https://github.com/user-attachments/assets/55144ca7-65e3-465c-bc7c-4631799baa7b" />

- **READ**
<img width="1383" height="741" alt="image" src="https://github.com/user-attachments/assets/57625602-f619-4df3-a369-60425fd4ab8b" />


Both PixelLM and READ are derived from the LISA architecture and incorporate mechanisms for region-level visual grounding. Among them, **READ** explicitly emphasizes *reasoning about where to attend* within an image.

---

### Research Focus

The central research question of this project is:

> Can the *reasoning capabilities of where to attend* in multimodal foundation models be effectively applied to anomaly detection, particularly for zero-shot anomaly segmentation?

By experimentally evaluating READ under training-free and zero-shot conditions, we aim to assess the feasibility of using multimodal foundation models as reasoning-centric anomaly detection systems rather than similarity-based detectors.

---

## 2. Pipeline

End-to-end inference pipeline used to evaluate **READ (Reasoning to Attend)** on the **MVTecAD anomaly segmentation** task in a **training-free, zero-shot** setting:

**MVTecAD test image**  
→ **CLIP vision tower (LLaVA)**  
→ **LLaVA-based multimodal LLM**  
→ **Similarity map + SasP point proposal**  
→ **SAM prompts (points + <SEG> token embedding)**  
→ **SAM mask decoder**  
→ **raw logits saved (.npy)**  
→ **benchmark metrics (iAUROC / pAUROC / PRO / F1-max)**

The key idea is that **CLIP-based visual features** and **LLM-derived textual features** jointly form the prompt conditioning for **SAM**, where:
- **point prompts** provide spatial anchors (where to segment),
- **<SEG> token embedding** provides semantic intent (what to segment),
and the final output is evaluated using standard MVTecAD segmentation metrics.

---

## 3. Architecture Walkthrough

These are how READ connects **LLaVA (CLIP + LLM)** and **SAM** into a single anomaly segmentation pipeline.

### 3.1 CLIP Vision Tower (LLaVA)

**Vision backbone:** `clip-vit-large-patch14-336`

Key configuration:
- input resolution: **336×336**
- patch size: **14**
- patch grid: `(336/14) × (336/14) = 24×24`
- number of patch tokens: **576**

Preprocessing (in code):
- `ImageProcessor.load_and_preprocess_image()`  
  → `CLIPImageProcessor.preprocess()`  
  → resize + center crop to **336×336**

This produces dense patch embeddings that become the basis for:
- cross-modal fusion in LLaVA,
- similarity computation against the `<SEG>` embedding.

---

### 3.2 LLaVA / READ: Multimodal Fusion and <SEG> Embedding

A core assumption in READ is that the model can produce an embedding that represents **“the region to be segmented”** in a way that is aligned with visual patch features.

**Input format**
- prompt template (representative): `Segment [ANOMALY_TYPE] on the [OBJECT].`
- image tokens + text tokens are concatenated as LLaVA-style multimodal input.

**Outputs used for segmentation**
- the multimodal LLM produces a **<SEG> token embedding**, which is treated as a semantic prompt for SAM.

---

### 3.3 SasP (Similarity as Points): Point Prompt Generation

READ uses a point-based prompting strategy to guide SAM.

1) A **similarity map** is computed between:
- `<SEG>` embedding (text side)
- CLIP patch embeddings (image side)

2) The pipeline extracts high-response regions and proposes point candidates.

3) Discrete-to-Continuous correction is applied:
- `Discrete_to_Continuous` refines point prompt coordinates into continuous space.

The result is a small set of points:
- `point_coords` (continuous coordinates)
- `point_labels` (foreground/background-like indicators for SAM prompt encoding)

These points become spatial “hints” that specify where SAM should segment.

---

### 3.4 SAM Prompt Encoder and Mask Decoder

**SAM input resolution:** fixed at **1024×1024**
- `ResizeLongestSide(1024)` + padding → 1024×1024 tensor
- SAM encoder patch size: **16**
- patch grid: **64×64**

SAM receives two prompt modalities:
- **point prompts**: `(point_coords, point_labels)`
- **text prompt**: `<SEG>` embedding (from LLaVA/READ)

Flow:
- `PromptEncoder` encodes points + embedding prompts
- `MaskDecoder` predicts segmentation masks (logits)
- `postprocess_masks()` restores masks to the original image resolution

---

## 4. Benchmark Metrics

We compute standard Industrial Anomaly Detection metrics:

- **iAUROC** (image-level AUROC)
- **pAUROC** (pixel-level AUROC)
- **PRO** (per-region overlap)
- **F1-max** (max F1 over threshold sweep)
- *(optional)* **PRO_hard** (more conservative region overlap variant; computationally heavier)

---

## 6. Qualitative Analysis: What Failed and Why

### 6.1 Prompt Design Constraints
We use a simple template:

`Segment [ANOMALY_TYPE] on the [OBJECT].`

This has inherent limitations:
- anomaly types may not be well-defined by a short phrase,
- some categories require extra reference or attribute grounding.

We also exclude cases where anomaly semantics are ambiguous or overly generic(e.g.,`*_color`, `*_combined` types in some categories).

---

### 6.2 Failure Cases That Require Additional Context
Some anomaly types consistently fail because the prompt alone cannot define the defect boundary or the “normal reference”:

Examples:
- `cable__cut_inner_insulation`
- `cable__missing_wire`
- `cable__poke_insulation`
- `metal_nut__bent`
- `tile__rough`
- `transistor__misplaced` (GT ambiguity can also exist)
- `tile__gray_stroke`
- `screw__scratch_head` (positional bias: model often assumes “head” is at the top)

In these cases, performance degradation suggests:
- the model needs either a richer description,
- or a reference image / explicit normal-state definition.

---

### 6.3 Primary Bottleneck: Unstable <SEG>–Patch Similarity (LLaVA side)
Across many categories, the dominant failure mode is not SAM decoding, but unstable activation in the similarity map formed by: `<SEG> embedding` × `image patch embeddings`

This yields:
- diffuse activations (poor localization)
- spurious responses to background or object texture
- inconsistent focus even under the same prompt template

As a result, even when the prompting logic is correct,
the segmentation target itself becomes unstable.

<img width="2094" height="1751" alt="llava_bad" src="https://github.com/user-attachments/assets/da21a11e-5b6a-4873-b1e0-ad6ad565d0b0" />

---

### 6.4 Secondary Bottlenecks: Point Prompt / SAM Decoding
Less frequently, the failure arises after similarity map formation:

- **Point prompt instability**  
  similarity activation is meaningful, but point extraction/refinement fails  
  <img width="2094" height="915" alt="point_bad" src="https://github.com/user-attachments/assets/5fe2af21-a946-431a-bd64-aadb4a1e4971" />


- **SAM decoding incompleteness**  
  points are valid, but the decoded mask is partial or misses structure  
  <img width="2095" height="914" alt="sam_bad" src="https://github.com/user-attachments/assets/62a19fe3-971e-4ae1-af74-0af8b6540083" />

---

### 6.5 Best Cases (All Components Align)
When `<SEG>` embedding is stable, points are well-formed, and SAM decodes correctly, READ can produce convincing anomaly localization even in a training-free setting.

<img width="2093" height="3008" alt="Segmentation Results" src="https://github.com/user-attachments/assets/515578de-7c8c-48c6-903b-8c2f24c1afa4" />


> Note: Since benchmark performance is not strong overall in this setup,
> highly granular case-by-case categorization may not always yield consistent conclusions.
> Nevertheless, qualitative breakdown helps identify the dominant bottleneck.

---

## 7. Final Takeaways

This project evaluates whether a segmentation-capable multimodal foundation pipeline (**LLaVA + READ + SAM**) can function as a reasoning-centric anomaly localization system under a **training-free, zero-shot** setting.

Key conclusions:
- **Training-free, zero-shot anomaly segmentation remains challenging.**
- The pipeline outputs **logits**, so anomaly scoring requires additional design
  (e.g., sigmoid/softmax normalization and threshold sweep).
- A more conservative region metric like **PRO_hard** may be useful when score distributions collapse,
  although it increases computation.
- Prompt choice matters:
  - asking only *“Is there any anomaly?”* yields poor performance overall.
  - specifying both **anomaly type + object** provides minor improvements but is not sufficient.
- While multiple factors can contribute (LLaVA `<SEG>`, READ point prompts, SAM decoding),
  the **dominant bottleneck is often the instability of `<SEG>` embedding and its similarity map.**

Potential next steps:
- stronger prompt engineering (descriptive attributes, constraints)
- incorporating reference images or normal-state descriptions
- few-shot adaptation or lightweight finetuning
- hyperparameter tuning (point selection, thresholding, score calibration)
- anomaly synthesis + training (for controlled supervision)
- exploring normal few-shot settings (instead of strict training-free)

---

## Supplementary for Code Implementation

### NOT included
- MVTec AD dataset
- Model weights / checkpoints
- Inference outputs (`read_mvtec_outputs*`)
- Cached environments (`.venv`, `__pycache__`, etc.)

### Conda & pip
```bash
conda env create -f environment.yml
conda activate read
```
```bash
pip install -r requirements.txt
```

### 1) Run Inference (MVTecAD)
```bash
python eval/mvtec_batch_infer.py \
  --mvtec_root /path/to/MVTecAD \
  --out_dir ./read_mvtecAD_outputs \
  --classes bottle,hazelnut \
  --max_per_class -1 \
  --prompt_mode anomaly
```

This will write per-image subfolders containing:
- `pred_logits_*.npy` (raw logits)
- `metadata.json` (image path + prompt)
- visualization images (optional)

### 2) Run Evaluation
```bash
python eval/eval_metrics.py \
  --pred_root ./read_mvtecAD_outputs \
  --mvtec_root /path/to/MVTecAD \
  --out_csv mvtecAD_eval.csv
```

Output CSV columns:
- `scope`: `overall` or `category`
- `category`: `ALL` or category name
- `iAUROC`: image-level AUROC
- `pAUROC`: pixel-level AUROC
- `PRO`: dataset-level PRO-AUC (FPR ≤ 0.3)
- `PRO_hard`: only if `--pro_hard` is enabled
- `F1_max`: pixel-level max F1
- `num_images`, `num_anom_images`

### Optional: PRO_hard
`PRO_hard` is memory/time intensive. Enable only if needed:
```bash
python eval/eval_metrics.py \
  --pred_root ./read_mvtec_outputs_default \
  --mvtec_root /path/to/MVTecAD \
  --out_csv mvtec_eval_0120.csv \
  --pro_hard
```

#### Notes
- `eval_metrics.py` assumes `pred_logits_*.npy` are **raw logits** and applies `sigmoid` internally.
- Anomaly images without GT masks are **skipped** for evaluation.

