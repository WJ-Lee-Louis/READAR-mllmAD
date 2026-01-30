# AnomalyDetection-mLLM-READ

This repository contains the minimal code needed to run **MVTec AD inference** with READ and compute benchmark metrics
(iAUROC, pAUROC, PRO, F1-max). Large artifacts such as datasets, model weights, and inference outputs are intentionally
excluded from version control.

## What is included
- Inference and evaluation:
  - `eval/mvtec_batch_infer.py`
  - `eval/eval_metrics.py`
- Model code:
  - `model/`
  - `my_inference.py`
  - `utils.py`
- Environment:
  - `environment.yml`
  - `requirements.txt`

## What is NOT included
- MVTec AD dataset
- Model weights / checkpoints
- Inference outputs (`read_mvtec_outputs*`)
- Cached environments (`.venv`, `__pycache__`, etc.)

## Environment
Use either Conda or pip:

### Conda
```bash
conda env create -f environment.yml
conda activate read
```

### pip
```bash
pip install -r requirements.txt
```

## Dataset (MVTec AD)
Download and place the dataset in a local path, e.g.:
```
/path/to/MVTecAD
```
Expected structure:
```
MVTecAD/<class>/test/<defect>/<image>.png
MVTecAD/<class>/ground_truth/<defect>/<image>_mask.png
```

## Model Weights
Prepare READ and vision tower weights locally. Example placeholders:
- `READ-LLaVA-v1.5-7B-for-ReasonSeg-valset`
- `clip-vit-large-patch14-336`

Pass the correct paths via arguments when running inference.

## 1) Run Inference (MVTec)
```bash
python eval/mvtec_batch_infer.py \
  --mvtec_root /path/to/MVTecAD \
  --out_dir ./read_mvtec_outputs_default \
  --classes bottle,hazelnut \
  --max_per_class -1 \
  --prompt_mode anomaly
```

This will write per-image subfolders containing:
- `pred_logits_*.npy` (raw logits)
- `metadata.json` (image path + prompt)
- visualization images (optional)

## 2) Run Evaluation
```bash
python eval/eval_metrics.py \
  --pred_root ./read_mvtec_outputs_default \
  --mvtec_root /path/to/MVTecAD \
  --out_csv mvtec_eval_0120.csv
```

Output CSV columns:
- `scope`: `overall` or `class`
- `class`: `ALL` or class name
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

## Notes
- `eval_metrics.py` assumes `pred_logits_*.npy` are **raw logits** and applies `sigmoid` internally.
- Anomaly images without GT masks are **skipped** for evaluation.

