#!/usr/bin/env python3
"""
Batch inference for MVTec using READ.

This script loads the READ model and runs `visualize_case` from
`my_inference.py` for images under selected MVTec classes. It now
defaults to GPU execution (if available) so full test splits can be
processed once GPUs are free, while still allowing a CPU-only fallback.
"""
import argparse
import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from model.READ import load_pretrained_model_READ
from my_inference import visualize_case


def parse_args():
    parser = argparse.ArgumentParser(description="Batch infer MVTec subset with READ")
    parser.add_argument("--pretrained_model_path", default="./READ-LLaVA-v1.5-7B-for-ReasonSeg-valset", type=str)
    parser.add_argument("--vision_tower", default="./clip-vit-large-patch14-336", type=str)
    parser.add_argument("--mvtec_root", required=True, type=str, help="Path to MVTec root directory")
    parser.add_argument("--out_dir", default="./read_mvtec_outputs", type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--classes", default="bottle,hazelnut", type=str,
                        help="Comma-separated list of MVTec classes to run (default: small subset)")
    parser.add_argument("--max_per_class", type=int, default=-1,
                        help="Maximum #images to process per class (-1 uses the entire test split)")
    parser.add_argument(
        "--prompt_mode",
        default="anomaly",
        choices=["anomaly", "binary", "template"],
        help="Prompt strategy: anomaly uses class/anomaly, binary uses a fixed question, template uses --prompt_template",
    )
    parser.add_argument(
        "--prompt_template",
        default="Segment the {anomaly_type} on the {class_name}.",
        help="Template used when --prompt_mode=template (supports {class_name}, {anomaly_type})",
    )
    parser.add_argument("--vis_only", action="store_true",
                        help="Only produce visualizations (metadata is still saved, logits are skipped)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if GPUs are available")
    parser.add_argument(
        "--save_raw",
        dest="save_raw",
        action="store_true",
        help="Save raw pred_mask logits as .npy for later evaluation (default)",
    )
    parser.add_argument(
        "--no_save_raw",
        dest="save_raw",
        action="store_false",
        help="Disable saving raw pred_mask logits",
    )
    parser.set_defaults(save_raw=True)
    return parser.parse_args()


def build_prompt(args, class_name, anomaly_type):
    if args.prompt_mode == "binary":
        return "Is there any anomaly in the image?"
    if args.prompt_mode == "template":
        return args.prompt_template.format(
            class_name=class_name, anomaly_type=anomaly_type
        )
    return f"Segment the {anomaly_type} on the {class_name}."


def collect_images_for_class(mvtec_root, class_name):
    test_dir = os.path.join(mvtec_root, class_name, "test")
    if not os.path.isdir(test_dir):
        return []
    img_paths = []
    for anomaly_type in os.listdir(test_dir):
        full_dir = os.path.join(test_dir, anomaly_type)
        if not os.path.isdir(full_dir):
            continue
        # collect png/jpg
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            img_paths.extend(glob.glob(os.path.join(full_dir, ext)))
    return sorted(img_paths)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    load_kwargs = dict(
        model_path=args.pretrained_model_path,
        vision_tower=args.vision_tower,
        model_max_length=args.model_max_length,
    )
    if args.cpu:
        load_kwargs["device"] = "cpu"

    print("Loading READ model...")
    tokenizer, segmentation_lmm, vision_tower, _ = load_pretrained_model_READ(**load_kwargs)

    # Device handling (follow my_inference.py behaviour)
    use_cpu = args.cpu or not torch.cuda.is_available()
    if use_cpu:
        print("Using CPU")
        segmentation_lmm = segmentation_lmm.cpu().to(torch.float32)
        vision_tower = vision_tower.cpu().to(torch.float32)
    else:
        print("Using GPU")
        device = next(segmentation_lmm.parameters()).device
        if device.type == 'cpu':
            segmentation_lmm = segmentation_lmm.to(torch.bfloat16).cuda()
        else:
            if segmentation_lmm.dtype != torch.bfloat16:
                segmentation_lmm = segmentation_lmm.to(torch.bfloat16)

        device = next(vision_tower.parameters()).device
        if device.type == 'cpu':
            vision_tower = vision_tower.to(torch.bfloat16).cuda()
        else:
            if vision_tower.dtype != torch.bfloat16:
                vision_tower = vision_tower.to(torch.bfloat16)

    # prepare a minimal args namespace compatible with visualize_case
    run_args = SimpleNamespace()
    run_args.model_max_length = args.model_max_length
    run_args.vis_save_dir = args.out_dir
    run_args.image_size = args.image_size

    for cls in classes:
        print(f"\n== Processing class: {cls} ==")
        images = collect_images_for_class(args.mvtec_root, cls)
        if len(images) == 0:
            print(f"No images found for class {cls} under {args.mvtec_root}")
            continue

        subset = images if args.max_per_class < 0 else images[:args.max_per_class]
        for img_path in subset:
            # derive anomaly type from path
            parts = img_path.split(os.sep)
            # expect .../<class>/test/<anomaly_type>/<img>
            try:
                anomaly_type = parts[-2]
            except Exception:
                anomaly_type = "anomaly"

            prompt = build_prompt(args, cls, anomaly_type)
            case_name = f"{cls}__{anomaly_type}__{os.path.basename(img_path).split('.')[0]}"
            case_config = {"image_path": img_path, "prompt": prompt}

            try:
                # Produce visualizations using the existing helper
                visualize_case(case_name, case_config, segmentation_lmm, tokenizer, vision_tower, run_args)

                save_dir = os.path.join(args.out_dir, case_name)
                os.makedirs(save_dir, exist_ok=True)
                meta = {
                    "image_path": img_path,
                    "prompt": prompt,
                    "prompt_mode": args.prompt_mode,
                    "prompt_template": args.prompt_template if args.prompt_mode == "template" else None,
                    "class_name": cls,
                    "anomaly_type": anomaly_type,
                }
                with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
                    json.dump(meta, f, indent=2)

                # Optionally save raw logits produced by the READ model for external evaluation
                if args.save_raw and not args.vis_only:
                    from my_inference import (
                        ImageProcessor,
                        DEFAULT_IMAGE_TOKEN,
                        conversation_lib,
                        replace_image_tokens,
                        tokenize_and_pad,
                    )

                    img_processor = ImageProcessor(vision_tower.image_processor, run_args.image_size)
                    image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(img_path)

                    conv = conversation_lib.default_conversation.copy()
                    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    conversation_list = [conv.get_prompt()]
                    if getattr(segmentation_lmm.config, "mm_use_im_start_end", False):
                        conversation_list = replace_image_tokens(conversation_list)

                    input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding='left')

                    # prepare tensors (cpu or cuda if available)
                    images_clip = torch.stack([image_clip], dim=0)
                    images = torch.stack([image], dim=0)
                    images_clip = images_clip.to(segmentation_lmm.dtype)
                    images = images.to(segmentation_lmm.dtype)
                    if not use_cpu and next(segmentation_lmm.parameters()).is_cuda:
                        images_clip = images_clip.cuda(non_blocking=True)
                        images = images.cuda(non_blocking=True)
                        input_ids = input_ids.cuda(non_blocking=True)

                    # Run generation to get hidden states and output ids (same as evaluate)
                    with torch.no_grad():
                        outputs = segmentation_lmm.generate(
                            images=images_clip,
                            input_ids=input_ids,
                            max_new_tokens=run_args.model_max_length,
                            num_beams=1,
                            output_hidden_states=True,
                            return_dict_in_generate=True,
                            do_sample=False,
                            temperature=0.2,
                        )
                        output_hidden_states = outputs.hidden_states[-1]
                        output_ids = outputs.sequences

                        # build seg_token_mask and padding as in model.evaluate
                        seg_token_mask = output_ids[:, 1:] == segmentation_lmm.seg_token_idx
                        vision = segmentation_lmm.get_vision_tower()
                        num_tokens_per_image = vision.num_patches
                        padding_left = torch.zeros(
                            seg_token_mask.shape[0],
                            num_tokens_per_image - 1,
                            dtype=seg_token_mask.dtype,
                            device=seg_token_mask.device,
                        )
                        seg_token_mask = torch.cat([padding_left, seg_token_mask], dim=1)

                        assert len(segmentation_lmm.model.text_hidden_fcs) == 1
                        output_hidden_states = output_hidden_states.to(seg_token_mask.device)
                        pred_embeddings = segmentation_lmm.model.text_hidden_fcs[0](output_hidden_states)
                        pred_embeddings = pred_embeddings.to(seg_token_mask.device)
                        pred_embeddings = pred_embeddings[seg_token_mask]

                        seg_token_counts = seg_token_mask.int().sum(-1)
                        seg_token_offset = seg_token_counts.cumsum(-1)
                        seg_token_offset = torch.cat(
                            [torch.zeros(1).long().to(seg_token_mask.device), seg_token_offset], dim=0
                        )

                        # split pred_embeddings per image
                        pred_embeddings_ = []
                        for i_idx in range(len(seg_token_offset) - 1):
                            start_i, end_i = seg_token_offset[i_idx], seg_token_offset[i_idx + 1]
                            if seg_token_counts[i_idx] == 0:
                                pred_embeddings_.append(None)
                            else:
                                batch_pred_embeddings = pred_embeddings[start_i:end_i]
                                pred_embeddings_.append(batch_pred_embeddings)

                        # obtain visual embeddings
                        image_embeddings = segmentation_lmm.get_visual_embs(images)
                        # keep dtypes consistent for matmul in similarity/mask generation
                        target_dtype = image_embeddings.dtype
                        pred_embeddings_ = [
                            (pe.to(target_dtype) if pe is not None and pe.dtype != target_dtype else pe)
                            for pe in pred_embeddings_
                        ]

                        # prepare SaSP points consistent with the visualizations
                        points_list = None
                        labels_list = None
                        try:
                            similarity = segmentation_lmm.compute_similarity_map(
                                image_embeddings, pred_embeddings_
                            )
                            pts, labs = segmentation_lmm.similarity_map_to_points(
                                similarity[0], sam_mask_shape[1], t=0.8
                            )
                            points_list = [pts]
                            labels_list = [labs]
                        except Exception:
                            points_list = None
                            labels_list = None
                        if points_list is None or labels_list is None:
                            points_list = [torch.zeros((1, 0, 2), device=segmentation_lmm.device)]
                            labels_list = [torch.zeros((1, 0), dtype=torch.int, device=segmentation_lmm.device)]

                        # call generate_pred_masks to get raw logits
                        pred_masks_raw = segmentation_lmm.generate_pred_masks(
                            pred_embeddings_,
                            image_embeddings,
                            [sam_mask_shape],
                            image_path=None,
                            point_coords=points_list,
                            point_labels=labels_list,
                            masks_list=None,
                            conversation_list=None,
                        )

                        for k_idx, pm in enumerate(pred_masks_raw):
                            if pm is None:
                                continue
                            try:
                                arr = pm.detach().cpu().numpy()
                            except Exception:
                                arr = np.array(pm)
                            np.save(os.path.join(save_dir, f"pred_logits_{k_idx}.npy"), arr)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print("All done.")


if __name__ == '__main__':
    main()
