import argparse
import json
import os

import cv2
import numpy as np
from skimage import measure
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def find_gt_for_image(image_path, mvtec_root=None):
    # Try to infer ground-truth mask path from standard MVTec layout
    parts = image_path.split(os.sep)
    try:
        # assume .../<class>/test/<defect>/<img>
        class_idx = None
        for i in range(len(parts)):
            if parts[i] == 'test':
                class_name = parts[i - 1]
                defect = parts[i + 1]
                break
        if mvtec_root is None:
            mvtec_root = os.path.join(*parts[: parts.index(class_name)])
    except Exception:
        return None

    gt_dir = os.path.join(mvtec_root, class_name, 'ground_truth', defect)
    if not os.path.isdir(gt_dir):
        # sometimes structure is <class>/ground_truth/<defect>
        gt_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'ground_truth', defect)
        if not os.path.isdir(gt_dir):
            return None

    base = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if base in f]
    if len(candidates) == 0:
        return None
    # prefer exact base_mask.png
    for c in candidates:
        if os.path.basename(c).startswith(base) and c.endswith('_mask.png'):
            return c
    return candidates[0]


def find_class_and_defect(image_path):
    parts = image_path.split(os.sep)
    for i in range(len(parts)):
        if parts[i] == 'test' and i > 0 and i + 1 < len(parts):
            class_name = parts[i - 1]
            defect = parts[i + 1]
            return class_name, defect
    return None, None


def compute_pauroc(pred_map, gt_mask):
    # pred_map: 2D float scores
    # gt_mask: 2D binary
    x = pred_map.flatten()
    y = gt_mask.flatten().astype(np.int32)
    if np.all(y == 0) or np.all(y == 1):
        return float('nan')
    return roc_auc_score(y, x)


def compute_image_score(pred_map):
    # use max value as image anomaly score
    return float(pred_map.max())


def compute_max_f1(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    a = 2 * precision * recall
    b = precision + recall
    f1s = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    if f1s.size == 0:
        return float('nan'), float('nan')
    index = int(np.argmax(f1s))
    max_f1 = float(f1s[index])
    if thresholds.size == 0:
        threshold = float('nan')
    elif index >= thresholds.size:
        threshold = float(thresholds[-1])
    else:
        threshold = float(thresholds[index])
    return max_f1, threshold


def compute_pro_dataset(score_maps, gt_masks, num_thresholds=200, fpr_thresh=0.3):
    # PRO for MVTec AD: average per-GT-region overlap vs FPR, integrated up to FPR threshold.
    if len(score_maps) == 0:
        return float('nan')

    max_th = max(float(s.max()) for s in score_maps)
    min_th = min(float(s.min()) for s in score_maps)
    if max_th == min_th:
        return float('nan')

    thresholds = np.linspace(max_th, min_th, num_thresholds)
    fprs = []
    pros = []
    precomp = []
    total_normal = 0
    for score_map, gt_mask in zip(score_maps, gt_masks):
        gt_mask = gt_mask.astype(bool)
        neg = np.logical_not(gt_mask)
        total_normal += neg.sum()
        region_masks = []
        region_areas = []
        if gt_mask.any():
            gt_regions = measure.label(gt_mask, connectivity=2)
            region_ids = np.unique(gt_regions)
            region_ids = region_ids[region_ids != 0]
            for region_id in region_ids:
                region_mask = gt_regions == region_id
                region_masks.append(region_mask)
                region_areas.append(float(region_mask.sum()))
        precomp.append((neg, region_masks, region_areas))

    if total_normal == 0:
        return float('nan')

    for t in thresholds:
        pro_list = []
        fp_pixels = 0
        for score_map, (neg, region_masks, region_areas) in zip(score_maps, precomp):
            pred_bin = score_map >= t

            if region_masks:
                for region_mask, region_area in zip(region_masks, region_areas):
                    overlap = np.logical_and(pred_bin, region_mask).sum() / region_area
                    pro_list.append(overlap)

            fp_pixels += np.logical_and(pred_bin, neg).sum()

        pros.append(float(np.mean(pro_list)) if len(pro_list) > 0 else 0.0)
        fprs.append(fp_pixels / float(total_normal))

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    within = fprs <= fpr_thresh
    if not np.any(within):
        return float('nan')

    fprs_clip = fprs[within]
    pros_clip = pros[within]

    if fprs_clip[0] > 0.0:
        pro_at_zero = float(np.interp(0.0, fprs, pros))
        fprs_clip = np.insert(fprs_clip, 0, 0.0)
        pros_clip = np.insert(pros_clip, 0, pro_at_zero)

    if fprs_clip[-1] < fpr_thresh:
        pro_at_max = float(np.interp(fpr_thresh, fprs, pros))
        fprs_clip = np.append(fprs_clip, fpr_thresh)
        pros_clip = np.append(pros_clip, pro_at_max)

    return float(auc(fprs_clip, pros_clip) / fpr_thresh)


def compute_pro_hard(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve at all distinct score thresholds."""
    if isinstance(anomaly_maps, list):
        anomaly_maps = np.stack(anomaly_maps)
    if isinstance(ground_truth_maps, list):
        ground_truth_maps = np.stack(ground_truth_maps)

    if len(ground_truth_maps.shape) == 4:
        ground_truth_maps = ground_truth_maps[:, 0, :, :]
    if len(anomaly_maps.shape) == 4:
        anomaly_maps = anomaly_maps[:, 0, :, :]

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps), anomaly_maps[0].shape[0], anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert (
        shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max
    ), "Potential overflow when using np.cumsum(), consider using np.uint64."

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):
        gt_map = gt_map.astype(bool)

        labeled, n_components = measure.label(gt_map, connectivity=2, return_num=True)
        num_gt_regions += n_components

        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1.0 / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    if num_ok_pixels == 0 or num_gt_regions == 0:
        return np.array([]), np.array([])

    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    np.clip(fprs, a_min=None, a_max=1.0, out=fprs)
    np.clip(pros, a_min=None, a_max=1.0, out=pros)

    zero = np.array([0.0])
    one = np.array([1.0])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))


def compute_pro_hard_auc(score_maps, gt_masks, fpr_thresh=0.3):
    fprs, pros = compute_pro_hard(score_maps, gt_masks)
    if fprs.size == 0 or pros.size == 0:
        return float('nan')

    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    within = fprs <= fpr_thresh
    if not np.any(within):
        return float('nan')

    fprs_clip = fprs[within]
    pros_clip = pros[within]

    if fprs_clip[0] > 0.0:
        pro_at_zero = float(np.interp(0.0, fprs, pros))
        fprs_clip = np.insert(fprs_clip, 0, 0.0)
        pros_clip = np.insert(pros_clip, 0, pro_at_zero)

    if fprs_clip[-1] < fpr_thresh:
        pro_at_max = float(np.interp(fpr_thresh, fprs, pros))
        fprs_clip = np.append(fprs_clip, fpr_thresh)
        pros_clip = np.append(pros_clip, pro_at_max)

    return float(auc(fprs_clip, pros_clip) / fpr_thresh)


def compute_fl_max(pred_map, gt_mask, num_thresholds=100):
    """Compute the best (max) F1/Dice score across thresholds (MVTec naming: FL-max)."""
    gt_mask = gt_mask.astype(bool)
    if gt_mask.sum() == 0:
        return float('nan')
    max_f1, _ = compute_max_f1(gt_mask.flatten().astype(int), pred_map.flatten())
    return float(max_f1)


def load_pred_logits(pred_dir):
    # find any pred_logits_*.npy and combine via max across masks
    files = [f for f in os.listdir(pred_dir) if f.startswith('pred_logits') and f.endswith('.npy')]
    if len(files) == 0:
        return None
    processed = []
    for f in files:
        arr = np.load(os.path.join(pred_dir, f))
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim > 3:
            # flatten any potential extra axes into channel dimension
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
        processed.append(arr)
    stacked = np.concatenate(processed, axis=0)
    probs = sigmoid(stacked)
    agg = probs.max(axis=0)
    return agg


def compute_dataset_metrics(score_maps, gt_masks, gt_labels, compute_pro=True, compute_pro_hard=False):
    img_scores = np.asarray([compute_image_score(s) for s in score_maps], dtype=float)
    gt_labels = np.asarray(gt_labels, dtype=int)

    if np.all(gt_labels == 0) or np.all(gt_labels == 1):
        i_roc = float('nan')
    else:
        i_roc = float(roc_auc_score(gt_labels, img_scores))

    gt_flat = np.concatenate([g.reshape(-1) for g in gt_masks]).astype(int)
    score_flat = np.concatenate([s.reshape(-1) for s in score_maps]).astype(float)
    if np.all(gt_flat == 0) or np.all(gt_flat == 1):
        p_roc = float('nan')
    else:
        p_roc = float(roc_auc_score(gt_flat, score_flat))

    if np.all(gt_flat == 0) or np.all(gt_flat == 1):
        p_f1 = float('nan')
    else:
        p_f1, _ = compute_max_f1(gt_flat, score_flat)

    pro = compute_pro_dataset(score_maps, gt_masks) if compute_pro else float('nan')
    pro_hard = compute_pro_hard_auc(score_maps, gt_masks) if compute_pro_hard else float('nan')

    return {
        'iAUROC': i_roc,
        'pAUROC': p_roc,
        'PRO': pro,
        'PRO_hard': pro_hard,
        'F1_max': p_f1,
        'num_images': int(len(score_maps)),
        'num_anom_images': int(gt_labels.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_root', required=True, help='Directory with per-image subdirs containing pred_logits_*.npy and metadata.json')
    parser.add_argument('--mvtec_root', required=False, help='Path to MVTec root if inference needed to locate GT masks')
    parser.add_argument('--out_csv', default='eval_results.csv')
    parser.add_argument('--pro_hard', action='store_true', help='Enable PRO_hard computation (memory/time intensive)')
    args = parser.parse_args()

    rows = []
    per_class = {}
    for item in sorted(os.listdir(args.pred_root)):
        sub = os.path.join(args.pred_root, item)
        if not os.path.isdir(sub):
            continue
        meta_path = os.path.join(sub, 'metadata.json')
        if not os.path.exists(meta_path):
            print('Skipping', sub, 'no metadata')
            continue
        meta = json.load(open(meta_path))
        img_path = meta.get('image_path')
        pred_map = load_pred_logits(sub)
        if pred_map is None:
            print('No pred logits in', sub)
            continue

        class_name, defect = find_class_and_defect(img_path) if img_path is not None else (None, None)
        if class_name is None:
            class_name = 'unknown'
        if defect is None:
            print('No defect label for', img_path, '-> skipping')
            continue

        gt_label = 0 if defect == 'good' else 1
        gt_path = find_gt_for_image(img_path, args.mvtec_root) if img_path is not None else None
        if gt_label == 1 and (gt_path is None or not os.path.exists(gt_path)):
            print('Missing GT for anomaly image', img_path, '-> skipping')
            continue
        if gt_path is not None and os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print('Failed to read GT', gt_path)
                continue
            gt_mask = (gt_mask > 0).astype(np.uint8)
        else:
            gt_mask = np.zeros_like(pred_map, dtype=np.uint8)

        rows.append({
            'class': class_name,
            'gt_label': gt_label,
            'pred_map': pred_map,
            'gt_mask': gt_mask,
        })

        per_class.setdefault(class_name, {'pred_maps': [], 'gt_masks': [], 'gt_labels': []})
        per_class[class_name]['pred_maps'].append(pred_map)
        per_class[class_name]['gt_masks'].append(gt_mask)
        per_class[class_name]['gt_labels'].append(gt_label)

    if len(rows) == 0:
        print('No valid predictions found.')
        return

    all_pred_maps = [r['pred_map'] for r in rows]
    all_gt_masks = [r['gt_mask'] for r in rows]
    all_gt_labels = [r['gt_label'] for r in rows]

    summary_rows = []
    overall = compute_dataset_metrics(
        all_pred_maps,
        all_gt_masks,
        all_gt_labels,
        compute_pro=True,
        compute_pro_hard=args.pro_hard,
    )
    overall.update({'scope': 'overall', 'class': 'ALL'})
    summary_rows.append(overall)

    for class_name, data in sorted(per_class.items()):
        metrics = compute_dataset_metrics(
            data['pred_maps'], data['gt_masks'], data['gt_labels'], compute_pro=False, compute_pro_hard=False
        )
        metrics.update({'scope': 'class', 'class': class_name})
        summary_rows.append(metrics)

    import csv
    keys = ['scope', 'class', 'iAUROC', 'pAUROC', 'PRO', 'PRO_hard', 'F1_max', 'num_images', 'num_anom_images']
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print('Wrote', args.out_csv)


if __name__ == '__main__':
    main()
