# READAR: Can Multimodal Language Models Reason to Attend to Anomaly Regions?

This repository contains the minimal code needed to run inference with MVTecAD Dataset via READ(Reasoning to Attend) and compute benchmark metrics
(iAUROC, pAUROC, PRO, F1-max). Large artifacts such as datasets, model weights, and inference outputs are intentionally
excluded.

## Background & Motivation
<img width="913" height="387" alt="image" src="https://github.com/user-attachments/assets/987ceb31-9434-47f8-9ad2-80be8477fd31" />

### Unsupervised Anomaly Detection

In real-world industrial and manufacturing environments, collecting sufficient anomaly samples is inherently difficult due to data scarcity and high acquisition costs. As a result, many practical anomaly detection settings assume that anomaly samples are unavailable during training, while test sets contain a mixture of normal and anomalous instances.

This constraint has led to extensive research on **unsupervised anomaly detection**, where models are trained exclusively on normal data and are expected to identify deviations at inference time. Such approaches have demonstrated robust and reliable performance across various industrial benchmarks.

---

### Taxonomy of Unsupervised Anomaly Detection Methods

Unsupervised anomaly detection methods can be broadly categorized into two main paradigms:

#### 1. Feature-Embedding-Based Methods

Feature-embedding-based approaches model the distribution of normal data in a learned feature space and detect anomalies based on deviations from this distribution. Representative techniques include: teacher-student architecture, one-class classification, distribution modeling(e.g., normalizaing flow-based), and feature memory bank-based. Among these, **feature memory bank approaches** such as **SPADE**, **PaDiM**, and **PatchCore** have achieved particularly strong **full-shot** performance. In some benchmarks, these methods demonstrate near-saturated performance, indicating the maturity and effectiveness of feature-based unsupervised anomaly detection.

#### 2. Reconstruction-Based Methods

Reconstruction-based methods learn to reconstruct normal images and use reconstruction errors as anomaly scores. This category includes approaches based on: autoencoder, GAN, transformer, diffusion. Despite their conceptual simplicity, reconstruction-based methods often face challenges related to reconstruction fidelity, training stability, and precise anomaly localization, especially for high-resolution industrial imagery.

---

### Vision–Language Models for Anomaly Detection

Recently, **Vision–Language Models (VLMs)** such as CLIP have been introduced into anomaly detection research. These methods leverage the alignment between **textual descriptions of normal/anomalous states** and **image representations** to enable **few-shot** or **zero-shot anomaly detection**: **WinCLIP**, **AnomalyCLIP**, etc. These approaches significantly improve data efficiency by exploiting pre-trained vision–language alignment. However, they primarily frame anomaly detection as a similarity-based scoring problem, where reasoning and semantic understanding play only a limited role.

---

### Multimodal LLMs for Anomaly Detection

More recent efforts aim to apply **multimodal Large Language Models (LLMs)** directly to anomaly detection tasks. A notable example is **AnomalyGPT**, which explores the integration of multimodal LLMs into anomaly detection pipelines.

The motivation for adopting multimodal LLMs in anomaly detection can be summarized as follows:
> 1. **Generalization and Adaptability**  
   The ability to generalize to unseen products, defect types, and domains without retraining.
> 2. **Explainability and Interaction**  
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

## 










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
- `scope`: `overall` or `class`
- `category`: `ALL` or class name
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

