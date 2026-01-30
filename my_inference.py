"""
READ 모델을 사용한 인퍼런스 및 시각화 스크립트

이 스크립트는 READ 모델을 사용하여:
1. Similarity map과 points를 계산하고 시각화
2. Segmentation mask를 생성하고 저장

저장되는 파일:
- similarity_map_with_points.jpg: Similarity map과 points 시각화
- similarity_map.jpg: Similarity map만
- seg_mask.jpg: Segmentation mask (grayscale)
- seg_rgb.jpg: 원본 이미지 위에 segmentation 결과 오버레이
- points_info.txt: Points 정보

사용 방법:
    python my_inference.py [옵션]

필수 옵션:
    --pretrained_model_path: READ 모델 경로
    --vision_tower: CLIP vision tower 경로
    --vis_save_dir: 결과 저장 디렉토리

예시:
    python my_inference.py \\
        --pretrained_model_path "./READ-LLaVA-v1.5-7B-for-ReasonSeg-valset" \\
        --vision_tower "./clip-vit-large-patch14-336" \\
        --vis_save_dir "./demo_directory"
"""
import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from model.READ import load_pretrained_model_READ
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad


def parse_args(args):
    parser = argparse.ArgumentParser(description="Visualize similarity maps, points, and segmentation masks for READ demo cases")
    parser.add_argument("--vision_tower", default="./clip-vit-large-patch14-336", type=str)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--pretrained_model_path", default="./READ-LLaVA-v1.5-7B-for-ReasonSeg-valset", type=str)
    parser.add_argument("--vis_save_dir", default="./demo_directory", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)


# 각 케이스의 정보 정의
CASE_CONFIGS = {
    # Bottle cases
    "bootle_broken_large": {  # Note: directory name has typo "bootle"
        "image_path": "/home/work/BAIKLAB/datasets/MVTecAD/bottle/test/broken_large/000.png",
        "prompt": "Segment the broken area on the bottle"
    },
    "bottle_broken_small": {
        "image_path": "/home/work/BAIKLAB/datasets/MVTecAD/bottle/test/broken_small/000.png",
        "prompt": "Segment the broken area on the bottle"
    },
    "bottle_contamination": {
        "image_path": "/home/work/BAIKLAB/datasets/MVTecAD/bottle/test/contamination/000.png",
        "prompt": "Segment the contamination on the bottle"
    },
    # Hazelnut cases
    "hazlenut_hole": {
        "image_path": "/home/work/BAIKLAB/datasets/MVTecAD/hazelnut/test/hole/000.png",
        "prompt": "Segment the hole on the hazelnut"
    },
    "hazlenut_print": {
        "image_path": "/home/work/BAIKLAB/datasets/MVTecAD/hazelnut/test/print/000.png",
        "prompt": "Segment the print on the hazelnut"
    },
    "hazlenut_crack": {
        "image_path": "/home/work/BAIKLAB/datasets/MVTecAD/hazelnut/test/crack/000.png",
        "prompt": "Segment the crack on the hazelnut"
    },
    "hazlenut_cut": {
        "image_path": "/home/work/BAIKLAB/datasets/MVTecAD/hazelnut/test/cut/000.png",
        "prompt": "Segment the cut on the hazelnut"
    },
}


def get_similarity_map(sm, shape):
    """similarity map을 이미지 크기로 변환"""
    # min-max norm
    sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
    # reshape
    side = int(sm.shape[1] ** 0.5)  # square output
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    # interpolate
    sm = sm.to(torch.float32)

    target_size = 336
    h, w = shape
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    sm = torch.nn.functional.interpolate(sm, (target_size, target_size), mode='bilinear')
    pad_h = (new_h - target_size) // 2
    pad_w = (new_w - target_size) // 2
    padded_sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h))
    sm = torch.nn.functional.interpolate(padded_sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm


@torch.inference_mode()
def extract_similarity_and_points(segmentation_lmm, image_clip, image, input_ids, sam_mask_shape, 
                                   default_im_start_token_idx, num_patches):
    """READ 모델에서 similarity map과 points를 추출"""
    # Forward pass
    output_ids, pred_masks, object_presence = segmentation_lmm.evaluate(
        image_clip.unsqueeze(0),
        image.unsqueeze(0),
        input_ids,
        [sam_mask_shape],
        max_new_tokens=512,
    )
    
    # Hidden states를 얻기 위해 다시 forward (similarity map 추출을 위해)
    # 실제로는 evaluate 내부에서 similarity를 계산하므로, 
    # evaluate 메서드를 수정하거나 hook을 사용해야 함
    # 여기서는 evaluate 내부 로직을 재현
    
    # Vision tower를 통해 이미지 특징 추출
    with torch.no_grad():
        # 이미지 특징 추출
        vision_outputs = segmentation_lmm.get_vision_tower()(
            image_clip.unsqueeze(0).to(segmentation_lmm.device)
        )
        image_features = vision_outputs[0]  # [1, num_patches, hidden_size]
        
        # Text 특징 추출을 위해 forward pass
        # 실제로는 evaluate 내부에서 계산되므로, 
        # 여기서는 간단히 모델의 forward를 호출하여 hidden states를 얻음
        outputs = segmentation_lmm.model(
            input_ids=input_ids.to(segmentation_lmm.device),
            images=image_clip.unsqueeze(0).to(segmentation_lmm.device),
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]  # 마지막 레이어의 hidden states
        
        # SEG token 위치 찾기
        seg_token_id = segmentation_lmm.config.seg_token_id if hasattr(segmentation_lmm.config, 'seg_token_id') else None
        if seg_token_id is None:
            # SEG token을 찾기 위해 다른 방법 사용
            # 일반적으로 특정 token ID를 사용
            seg_token_mask = (input_ids == 32000)  # READ에서 사용하는 SEG token ID 추정
        
        # Image embedding tokens와 SEG embedding tokens 추출
        seg_token_counts = seg_token_mask.int().sum(-1)
        image_embedding_tokens = hidden_states[seg_token_counts == 1]
        seg_embedding_tokens = hidden_states[seg_token_mask]
        
        # Similarity map 계산
        similarity = segmentation_lmm.compute_similarity_map(
            image_embedding_tokens[:, default_im_start_token_idx + 1:default_im_start_token_idx + 1 + num_patches, :],
            seg_embedding_tokens
        )
        
        # Points 추출
        points, labels = segmentation_lmm.similarity_map_to_points(
            similarity[0, :, 0], 
            sam_mask_shape[1], 
            t=0.8
        )
        
        # Similarity map을 이미지 크기로 변환
        similarity_map = get_similarity_map(similarity, sam_mask_shape[1])
        
    return similarity_map, points, labels, pred_masks[0]


def visualize_similarity_map_and_points(image_path, similarity_map, points, labels, save_path):
    """Similarity map과 points를 시각화"""
    # 원본 이미지 로드
    cv2_img = cv2.imread(image_path)
    if cv2_img is None:
        pil_img = Image.open(image_path)
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    h, w = cv2_img.shape[:2]
    
    # Similarity map을 numpy로 변환
    sm_vis = (similarity_map[0, ..., 0].detach().cpu().numpy() * 255).astype('uint8')
    
    # Similarity map을 컬러맵으로 변환
    sm_colored = cv2.applyColorMap(sm_vis, cv2.COLORMAP_JET)
    
    # 원본 이미지와 similarity map을 블렌딩
    vis = cv2_img * 0.3 + sm_colored * 0.7
    
    # Points 그리기
    for i, pt in enumerate(points):
        # Handle both numpy array and tensor formats
        if isinstance(pt, (list, tuple)):
            x, y = int(pt[0]), int(pt[1])
        elif isinstance(pt, np.ndarray):
            x, y = int(pt[0]), int(pt[1])
        else:  # tensor
            x, y = int(pt[0].item()), int(pt[1].item())
        
        # Handle label format
        if isinstance(labels, np.ndarray):
            label_val = labels[i]
        elif isinstance(labels, torch.Tensor):
            label_val = labels[i].item()
        else:
            label_val = labels[i]
        
        # Positive points (label=1)는 파란색, Negative points (label=0)는 빨간색
        color = (0, 0, 255) if label_val == 1 else (255, 0, 0)
        cv2.circle(vis, (x, y), 5, (255, 255, 255), 3)  # 흰색 외곽선
        cv2.circle(vis, (x, y), 3, color, -1)  # 채워진 원
    
    # BGR to RGB
    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    
    # 저장
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to: {save_path}")
    
    return vis


def save_segmentation_mask(pred_mask, image_path, save_dir):
    """Segmentation mask를 저장 (mask만, 그리고 이미지 위에 오버레이)"""
    # pred_mask를 numpy로 변환
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    
    # Shape 처리: [1, H, W] 또는 [H, W] 형태일 수 있음
    if len(pred_mask.shape) == 3:
        if pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]  # [1, H, W] -> [H, W]
        else:
            pred_mask = pred_mask[0]  # [C, H, W] -> [H, W] (첫 번째 채널 사용)
    elif len(pred_mask.shape) == 2:
        pass  # 이미 [H, W] 형태
    else:
        raise ValueError(f"Unexpected pred_mask shape: {pred_mask.shape}")
    
    # Binary mask로 변환
    pred_mask = pred_mask > 0
    
    # 원본 이미지 로드
    image_np = cv2.imread(image_path)
    if image_np is None:
        pil_img = Image.open(image_path)
        image_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Mask 크기가 이미지와 다를 수 있으므로 리사이즈
    if pred_mask.shape[:2] != image_np.shape[:2]:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                              (image_np.shape[1], image_np.shape[0]), 
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # 1. Segmentation mask만 저장 (grayscale)
    seg_mask_path = os.path.join(save_dir, "seg_mask.jpg")
    seg_mask_vis = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(seg_mask_path, seg_mask_vis)
    print(f"Saved segmentation mask to: {seg_mask_path}")
    
    # 2. 이미지 위에 segmentation 결과 오버레이
    seg_rgb_path = os.path.join(save_dir, "seg_rgb.jpg")
    seg_rgb_vis = image_np.copy()
    # Mask 영역에 파란색 오버레이 (demo.py의 방식과 동일)
    blue_color = np.array([0, 0, 255], dtype=np.uint8)  # BGR 형식
    # pred_mask가 True인 픽셀에만 적용
    mask_indices = pred_mask
    seg_rgb_vis[mask_indices] = (
        image_np[mask_indices] * 0.3 + blue_color * 0.7
    ).astype(np.uint8)
    cv2.imwrite(seg_rgb_path, seg_rgb_vis)
    print(f"Saved segmentation overlay to: {seg_rgb_path}")
    
    return seg_mask_path, seg_rgb_path


@torch.inference_mode()
def evaluate_with_similarity(segmentation_lmm, images_clip, images, input_ids, sam_mask_shape_list, max_new_tokens=32):
    """evaluate 메서드를 수정하여 similarity map과 points도 반환"""
    outputs = segmentation_lmm.generate(
        images=images_clip,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        output_hidden_states=True,
        return_dict_in_generate=True,
        do_sample=False,
        temperature=0.2
    )

    output_hidden_states = outputs.hidden_states[-1]
    output_ids = outputs.sequences

    seg_token_mask = output_ids[:, 1:] == segmentation_lmm.seg_token_idx

    # HACK: padding numer-of-token-per-image in total 
    vision_tower = segmentation_lmm.get_vision_tower()
    num_tokens_per_image = vision_tower.num_patches
    padding_left = torch.zeros(
        seg_token_mask.shape[0],
        num_tokens_per_image - 1,
        dtype=seg_token_mask.dtype,
        device=seg_token_mask.device,
    )
    seg_token_mask = torch.cat(
        [padding_left, seg_token_mask],
        dim=1,
    )
    assert len(segmentation_lmm.model.text_hidden_fcs) == 1
    output_hidden_states = output_hidden_states.to(seg_token_mask.device)
    pred_embeddings = segmentation_lmm.model.text_hidden_fcs[0](output_hidden_states)
    pred_embeddings = pred_embeddings.to(seg_token_mask.device)
    pred_embeddings = pred_embeddings[seg_token_mask]

    seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

    seg_token_offset = seg_token_counts.cumsum(-1)
    seg_token_offset = torch.cat(
        [torch.zeros(1).long().to(seg_token_mask.device), seg_token_offset],
        dim=0,
    )

    pred_embeddings_ = []
    object_presence = []
    for i in range(len(seg_token_offset) - 1):
        if seg_token_counts[i] == 0:
            pred_embeddings_.append(None)
            object_presence.append(False)
        else:
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
            object_presence.append(True)
   
    pred_embeddings = pred_embeddings_

    # Run SAM
    image_embeddings = segmentation_lmm.get_visual_embs(images)
    
    image_embedding_tokens = output_hidden_states[seg_token_counts==1]
    seg_embedding_tokens = output_hidden_states[seg_token_mask]
    default_im_start_token_idx = torch.where(input_ids==32001)[1][0].item()
    points_list, labels_list = [], []
    similarity_list = []  # Similarity maps 저장
    
    for bs in range(len(image_embedding_tokens)):
        similarity = segmentation_lmm.compute_similarity_map(
            image_embedding_tokens[ 
                bs: bs+1, 
                default_im_start_token_idx + 1: default_im_start_token_idx + 1 \
                + segmentation_lmm.get_vision_tower().num_patches, :
            ],
            seg_embedding_tokens[bs: bs + 1, ...]
        )
        similarity_list.append(similarity)
        points1, labels1 = segmentation_lmm.similarity_map_to_points(similarity[0, :, 0], sam_mask_shape_list[0][1], t=0.8)
        points_list.append(points1[None,...])
        labels_list.append(labels1[None,...])

    pred_masks = segmentation_lmm.generate_pred_masks(
        pred_embeddings, 
        image_embeddings, 
        sam_mask_shape_list, 
        image_path=None,
        point_coords=points_list,
        point_labels=labels_list,
        masks_list=None,
        conversation_list=None
    )  
    
    # Post processing for inference
    output_pred_masks = []
    for i, pred_mask in enumerate(pred_masks):
        if pred_embeddings[i] is not None:
            pred_mask = (pred_mask[0] > 0).int()
            if pred_mask.sum() == 0:
                object_presence[i] = False
            output_pred_masks.append(pred_mask)
        else:
            output_pred_masks.append(pred_mask)
    
    return output_ids, output_pred_masks, object_presence, similarity_list, points_list, labels_list


@torch.inference_mode()
def visualize_case(case_name, case_config, segmentation_lmm, tokenizer, vision_tower, args):
    """단일 케이스에 대해 similarity map, points, 그리고 segmentation mask를 시각화"""
    print(f"\n{'='*70}")
    print(f"Processing case: {case_name}")
    print(f"Image: {case_config['image_path']}")
    print(f"Prompt: {case_config['prompt']}")
    print(f"{'='*70}")
    
    # 이미지 로드 및 전처리
    img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
    image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(case_config['image_path'])
    
    # 질문 포맷팅
    conv = conversation_lib.default_conversation.copy()
    question = DEFAULT_IMAGE_TOKEN + "\n" + case_config['prompt']
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    conversation_list = [conv.get_prompt()]
    
    mm_use_im_start_end = getattr(segmentation_lmm.config, "mm_use_im_start_end", False)
    if mm_use_im_start_end:
        conversation_list = replace_image_tokens(conversation_list)
    
    input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding='left')
    
    # Input dictionary 준비
    use_cpu = not torch.cuda.is_available()
    if use_cpu or segmentation_lmm.dtype == torch.float32:
        precision = "fp32"
        is_cuda = False
    else:
        precision = "bf16"
        is_cuda = True
    
    input_dict = {
        "image_path": case_config['image_path'],
        "images_clip": torch.stack([image_clip], dim=0),
        "images": torch.stack([image], dim=0),
        "input_ids": input_ids,
        "sam_mask_shape_list": [sam_mask_shape]
    }
    input_dict = prepare_input(input_dict, precision, is_cuda=is_cuda)
    
    # 입력을 cuda:0으로 명시적으로 이동 (accelerate가 자동으로 분산 처리)
    # 모델이 여러 GPU에 분산되어 있어도 입력은 첫 번째 GPU로 보냄
    if is_cuda and torch.cuda.is_available():
        input_dict["images_clip"] = input_dict["images_clip"].cuda(non_blocking=True)
        input_dict["images"] = input_dict["images"].cuda(non_blocking=True)
        input_dict["input_ids"] = input_dict["input_ids"].cuda(non_blocking=True)
    
    # Similarity map과 points 추출
    try:
        output_ids, pred_masks, object_presence, similarity_list, points_list, labels_list = evaluate_with_similarity(
            segmentation_lmm,
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )
        
        if len(similarity_list) > 0:
            similarity = similarity_list[0]
            points_tensor = points_list[0][0]  # Remove batch dimension: [num_points, 2]
            labels_tensor = labels_list[0][0]  # Remove batch dimension: [num_points]
            
            # Convert tensors to numpy for visualization
            points = points_tensor.detach().cpu().numpy() if isinstance(points_tensor, torch.Tensor) else points_tensor
            labels = labels_tensor.detach().cpu().numpy() if isinstance(labels_tensor, torch.Tensor) else labels_tensor
            
            # Similarity map을 이미지 크기로 변환
            similarity_map = get_similarity_map(similarity, sam_mask_shape[1])
            
            # 시각화 저장 경로
            save_dir = os.path.join(args.vis_save_dir, case_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "similarity_map_with_points.jpg")
            
            # 시각화
            vis = visualize_similarity_map_and_points(
                case_config['image_path'],
                similarity_map,
                points,
                labels,
                save_path
            )
            
            # Similarity map만 따로 저장
            sm_vis = (similarity_map[0, ..., 0].detach().cpu().numpy() * 255).astype('uint8')
            sm_colored = cv2.applyColorMap(sm_vis, cv2.COLORMAP_JET)
            sm_save_path = os.path.join(save_dir, "similarity_map.jpg")
            cv2.imwrite(sm_save_path, sm_colored)
            print(f"Saved similarity map to: {sm_save_path}")
            
            # Points 정보를 텍스트로 저장
            points_info_path = os.path.join(save_dir, "points_info.txt")
            with open(points_info_path, 'w') as f:
                f.write(f"Total points: {len(points)}\n")
                positive_count = (labels == 1).sum() if isinstance(labels, np.ndarray) else (labels == 1).sum().item()
                negative_count = (labels == 0).sum() if isinstance(labels, np.ndarray) else (labels == 0).sum().item()
                f.write(f"Positive points (label=1): {positive_count}\n")
                f.write(f"Negative points (label=0): {negative_count}\n")
                f.write(f"\nPoints (x, y, label):\n")
                for i, (pt, lbl) in enumerate(zip(points, labels)):
                    x_val = pt[0] if isinstance(pt, np.ndarray) else pt[0].item()
                    y_val = pt[1] if isinstance(pt, np.ndarray) else pt[1].item()
                    lbl_val = lbl if isinstance(lbl, (int, np.integer)) else lbl.item()
                    f.write(f"{i}: ({x_val:.1f}, {y_val:.1f}), label={lbl_val}\n")
            print(f"Saved points info to: {points_info_path}")
            
            # Segmentation mask 저장
            if len(pred_masks) > 0 and pred_masks[0] is not None:
                save_segmentation_mask(
                    pred_masks[0],
                    case_config['image_path'],
                    save_dir
                )
            
            print(f"✓ Completed: {case_name}")
        else:
            print(f"✗ No similarity maps extracted for {case_name}")
    except Exception as e:
        print(f"✗ Error processing {case_name}: {e}")
        import traceback
        traceback.print_exc()


def main(args):
    args = parse_args(args)
    
    # CUDA device 설정은 환경 변수로 이미 설정되어 있을 수 있으므로 확인만 함
    # 모델이 여러 GPU에 분산되어 있을 수 있으므로 직접 device를 변경하지 않음
    
    print("="*70)
    print("READ Similarity Map, Points, and Segmentation Mask Visualization")
    print("="*70)
    
    # 모델 로드
    print("\nLoading READ model...")
    (
        tokenizer,
        segmentation_lmm,
        vision_tower,
        context_len,
    ) = load_pretrained_model_READ(
        model_path=args.pretrained_model_path,
        vision_tower=args.vision_tower,
        model_max_length=args.model_max_length
    )
    
    # Device 설정
    use_cpu = not torch.cuda.is_available()
    if use_cpu:
        print("⚠️  Using CPU mode")
        segmentation_lmm = segmentation_lmm.cpu().to(torch.float32)
        vision_tower = vision_tower.cpu().to(torch.float32)
    else:
        print("✓ Using GPU mode")
        # 모델이 이미 device에 있을 수 있으므로 확인 후 이동
        device = next(segmentation_lmm.parameters()).device
        if device.type == 'cpu':
            segmentation_lmm = segmentation_lmm.to(torch.bfloat16).cuda()
        else:
            # 이미 GPU에 있으면 dtype만 변경
            if segmentation_lmm.dtype != torch.bfloat16:
                segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
        
        device = next(vision_tower.parameters()).device
        if device.type == 'cpu':
            vision_tower = vision_tower.to(torch.bfloat16).cuda()
        else:
            if vision_tower.dtype != torch.bfloat16:
                vision_tower = vision_tower.to(torch.bfloat16)
    
    tokenizer.padding_side = "left"
    
    # 각 케이스에 대해 시각화
    for case_name, case_config in CASE_CONFIGS.items():
        try:
            visualize_case(case_name, case_config, segmentation_lmm, tokenizer, vision_tower, args)
        except Exception as e:
            print(f"✗ Error processing {case_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All visualizations completed!")
    print("="*70)


if __name__ == "__main__":
    main(sys.argv[1:])

