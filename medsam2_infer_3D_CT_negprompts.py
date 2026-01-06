from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename, exists
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import argparse
import json

from PIL import Image
import SimpleITK as sitk
import torch

# Import MedSAM2 package (assumes MedSAM2/ is a sibling directory)
import sys
sys.path.insert(0, 'MedSAM2')
import sam2  # initializes Hydra config module
from sam2.build_sam import build_sam2_video_predictor_npz


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    Input: array (D, H, W)
    Output: (D, 3, image_size, image_size)
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.uint8)
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array
    return resized_array


def sample_negative_points(negative_mask_2d, max_points=50):
    """
    Sample points from negative mask for negative prompts.
    More efficient than using all points.
    """
    neg_coords = np.argwhere(negative_mask_2d > 0.5)
    
    if len(neg_coords) == 0:
        return None
    
    # Sample points if too many
    if len(neg_coords) > max_points:
        indices = np.random.choice(len(neg_coords), max_points, replace=False)
        neg_coords = neg_coords[indices]
    
    # Return as (N, 2) array with [y, x] coordinates (for SAM2 format)
    return neg_coords[:, [1, 0]].astype(np.float32)  # [x, y] format


def run_medsam2_single_label_with_negatives(
    predictor,
    ct_np,            # (D, H, W) pre-windowed and rescaled to 0–255 uint8
    positive_mask_3d, # (D, H, W) binary mask - positive prompt
    negative_mask_3d, # (D, H, W) binary mask - negative prompt (other labels)
):
    """
    Per-slice 2D MedSAM2 refinement with negative prompts:
    - Find all slices where positive mask has non-zero voxels
    - For each slice:
      - Add positive mask as prompt (what to segment)
      - Add negative point prompts from negative mask (what NOT to segment)
    - Stack refined slices back into 3D
    Returns: segs_3D (D, H, W) binary mask
    """
    D, H, W = ct_np.shape
    segs_3D = np.zeros((D, H, W), dtype=np.uint8)

    # Find all slices where positive mask has content
    slices_with_mask = []
    for z in range(D):
        if np.any(positive_mask_3d[z] > 0.5):
            slices_with_mask.append(z)
    
    if len(slices_with_mask) == 0:
        print(f"    Warning: positive mask is completely empty, skipping")
        return segs_3D
    
    print(f"    Refining {len(slices_with_mask)} slices with negative prompts (z={slices_with_mask[0]}..{slices_with_mask[-1]})")

    # Prepare video tensor (all slices at once)
    video_height, video_width = H, W
    img_resized = resize_grayscale_to_rgb_and_resize(ct_np, 512)  # (D, 3, 512, 512)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = torch.from_numpy(img_resized).to(predictor.device)

    # Normalize as in other scripts
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].to(predictor.device)
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].to(predictor.device)
    img_resized = (img_resized - img_mean) / img_std

    # Refine each slice independently
    with torch.inference_mode():
        inference_state = predictor.init_state(
            images=img_resized,
            video_height=video_height,
            video_width=video_width,
        )
        
        for z in slices_with_mask:
            # Extract 2D masks for this slice
            pos_mask_2d = positive_mask_3d[z] > 0.5
            neg_mask_2d = negative_mask_3d[z] > 0.5
            
            if np.sum(pos_mask_2d) == 0:
                continue  # Skip if somehow empty

            # Reset state before each slice to ensure independence
            predictor.reset_state(inference_state)
            
            # Re-initialize state for this slice (needed after reset)
            inference_state = predictor.init_state(
                images=img_resized,
                video_height=video_height,
                video_width=video_width,
            )

            # Step 1: Add positive mask prompt (what to segment)
            frame_idx, obj_ids, masks_pos = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=z,
                obj_id=1,
                mask=pos_mask_2d,
            )

            # Step 2: Add negative point prompts (what NOT to segment)
            masks = masks_pos  # Default to positive mask result
            neg_points = sample_negative_points(neg_mask_2d, max_points=50)
            if neg_points is not None and len(neg_points) > 0:
                # Convert to torch tensors - ensure correct format [x, y] normalized
                points_tensor = torch.from_numpy(neg_points).to(predictor.device)  # (N, 2) [x, y]
                labels_tensor = torch.zeros(len(neg_points), dtype=torch.int32).to(predictor.device)  # 0 = negative
                
                # Add negative points - this should refine the mask to exclude these regions
                # The returned masks will have negative constraints applied
                frame_idx, obj_ids, masks_neg = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=z,
                    obj_id=1,
                    points=points_tensor.unsqueeze(0),  # (1, N, 2)
                    labels=labels_tensor.unsqueeze(0),  # (1, N)
                    clear_old_points=False,  # Keep the positive mask
                    normalize_coords=True,  # Normalize coordinates to [0,1]
                )
                # Use refined masks if they have content, otherwise fall back to positive mask
                if masks_neg is not None and len(masks_neg) > 0:
                    mask_check = masks_neg[0]
                    if mask_check.dim() == 3:
                        mask_check = mask_check.squeeze(0)
                    if (mask_check > 0.0).any().item():
                        masks = masks_neg

            # Extract refined mask: masks is [num_objects, H, W] at original video resolution
            if masks is None or len(masks) == 0:
                print(f"    Warning: No masks returned for slice {z}")
                predictor.reset_state(inference_state)
                continue
                
            mask_tensor = masks[0]  # Get first (and only) object's mask
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.squeeze(0)  # Remove channel dim if present
            
            # Convert to boolean numpy array at original resolution
            mask_refined_2d = (mask_tensor > 0.0).cpu().numpy()
            
            # Ensure it's 2D boolean
            if mask_refined_2d.ndim != 2:
                print(f"    Warning: Unexpected mask shape {mask_refined_2d.shape} for slice {z}")
                predictor.reset_state(inference_state)
                continue
            
            # Apply refined mask to this slice in the 3D volume
            # Use proper boolean indexing like in the working minimal script
            segs_3D[z][mask_refined_2d] = 1
            
            num_pixels = np.count_nonzero(mask_refined_2d)
            if num_pixels == 0:
                print(f"    Warning: Refined mask is empty for slice {z}")

            predictor.reset_state(inference_state)

    return segs_3D


def main():
    parser = argparse.ArgumentParser(
        description="MedSAM2 inference with positive and negative mask prompts"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='path to MedSAM2 checkpoint (e.g. MedSAM2/checkpoints/MedSAM2_CTLesion.pt)',
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help='MedSAM2 model config name (relative to sam2/configs/)',
    )
    parser.add_argument(
        '--prompts_dir',
        type=str,
        required=True,
        help='Directory containing prompt JSON files (from nnunet_to_medsam2_prompts_masks.py)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save refined masks (per label)',
    )

    args = parser.parse_args()

    checkpoint = args.checkpoint
    model_cfg = args.cfg
    prompts_dir = args.prompts_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Force CPU to avoid MPS/CUDA issues on macOS
    device = "cpu"
    print(f"Using device: {device} for MedSAM2 inference with negative prompts")

    predictor = build_sam2_video_predictor_npz(
        config_file=model_cfg,
        ckpt_path=checkpoint,
        device=device,
    )

    # Find case directories (new format: prompts_dir/case_id/prompts.json)
    case_dirs = []
    for item in os.listdir(prompts_dir):
        item_path = join(prompts_dir, item)
        if os.path.isdir(item_path):
            json_path = join(item_path, "prompts.json")
            if exists(json_path):
                case_dirs.append((item, json_path))
    
    # Also check for flat JSON files (backward compatibility)
    json_files = [
        f for f in os.listdir(prompts_dir)
        if f.endswith('.json') and not f.startswith('._') and os.path.isfile(join(prompts_dir, f))
    ]
    
    if case_dirs:
        print(f"Found {len(case_dirs)} case directories with prompts.json")
        use_new_format = True
    elif json_files:
        print(f"Found {len(json_files)} flat JSON files (old format)")
        use_new_format = False
    else:
        print(f"ERROR: No prompt files found in {prompts_dir}")
        return

    # Process cases
    if use_new_format:
        # Store both directory name and cleaned case_id
        cases_to_process = [(dir_name, json_path) for dir_name, json_path in case_dirs]
    else:
        cases_to_process = [(f.replace('.json', ''), join(prompts_dir, f)) for f in json_files]

    for dir_name, json_path in tqdm(cases_to_process, desc="Cases"):
        with open(json_path, 'r') as f:
            case_data = json.load(f)

        # Get case_id from JSON if available, and clean it up
        case_id = case_data.get('case_id', dir_name)
        # Remove any .nii or .nii.gz extensions that might have been included
        case_id_clean = case_id.replace('.nii.gz', '').replace('.nii', '')
        
        # Get image path - try multiple locations
        image_path = case_data.get('image_path')
        if not image_path or not exists(image_path):
            # Try to find CT image in common locations using cleaned case_id
            for pattern in [f"ct_images/{case_id_clean}_0000.nii.gz", f"ct_images/{case_id_clean}.nii.gz"]:
                if exists(pattern):
                    image_path = pattern
                    break
        
        if not image_path or not exists(image_path):
            print(f"[WARN] CT image not found for {case_id_clean}")
            continue

        print(f"\nProcessing case {case_id_clean}")

        # Load CT as numpy (D, H, W)
        ct_img = sitk.ReadImage(image_path)
        ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)

        # Simple CT windowing (soft tissue-ish)
        lower_bound, upper_bound = -160.0, 240.0
        ct_np = np.clip(ct_np, lower_bound, upper_bound)
        ct_np = (ct_np - ct_np.min()) / (ct_np.max() - ct_np.min() + 1e-8) * 255.0
        ct_np = ct_np.astype(np.uint8)

        # Get labels from new format
        labels_data = case_data.get('labels', {})
        if not labels_data:
            print(f"  [WARN] No labels found for {case_id_clean}")
            continue

        # Determine base directory for mask files
        # Use the original directory name (which might have .nii extension)
        if use_new_format:
            mask_base_dir = join(prompts_dir, dir_name)
        else:
            mask_base_dir = prompts_dir

        for label_name, label_info in labels_data.items():
            positive_mask_path = join(mask_base_dir, label_info['positive_mask_path'])
            negative_mask_path = join(mask_base_dir, label_info['negative_mask_path'])

            print(f"  Label {label_name}: pos={positive_mask_path}, neg={negative_mask_path}")

            if not exists(positive_mask_path):
                print(f"    [WARN] Positive mask not found: {positive_mask_path}")
                continue

            if not exists(negative_mask_path):
                print(f"    [WARN] Negative mask not found: {negative_mask_path}, using empty mask")
                negative_mask_path = None

            # Load positive mask
            pos_img = sitk.ReadImage(positive_mask_path)
            pos_np = sitk.GetArrayFromImage(pos_img).astype(np.uint8)

            # Load negative mask (or create empty if not found)
            if negative_mask_path and exists(negative_mask_path):
                neg_img = sitk.ReadImage(negative_mask_path)
                neg_np = sitk.GetArrayFromImage(neg_img).astype(np.uint8)
            else:
                neg_np = np.zeros_like(pos_np, dtype=np.uint8)

            # Check shape alignment
            if pos_np.shape != ct_np.shape:
                print(f"    [WARN] Shape mismatch CT {ct_np.shape} vs pos {pos_np.shape}, skipping")
                continue

            if neg_np.shape != ct_np.shape:
                print(f"    [WARN] Shape mismatch CT {ct_np.shape} vs neg {neg_np.shape}, using empty neg mask")
                neg_np = np.zeros_like(ct_np, dtype=np.uint8)

            # Sanity: ensure positive mask is not empty
            if np.count_nonzero(pos_np) == 0:
                print(f"    [WARN] Positive mask is completely empty for {label_name}, skipping")
                continue

            # Refine with positive and negative prompts
            segs_3D = run_medsam2_single_label_with_negatives(
                predictor=predictor,
                ct_np=ct_np,
                positive_mask_3d=pos_np,
                negative_mask_3d=neg_np,
            )

            # Save refined mask as binary (0/1) uint8
            mask_binary = (segs_3D > 0).astype(np.uint8)
            out_mask_img = sitk.GetImageFromArray(mask_binary)
            out_mask_img.CopyInformation(ct_img)

            # Use _negprompts suffix to distinguish from regular method
            save_name = f"{case_id_clean}_{label_name}_negprompts_mask.nii.gz"
            out_path = join(output_dir, save_name)
            sitk.WriteImage(out_mask_img, out_path)
            
            # Verify what we saved
            num_voxels = np.count_nonzero(mask_binary)
            print(f"    Saved refined mask with negative prompts: {out_path}")
            print(f"      Non-zero voxels: {num_voxels}, Unique values: {np.unique(mask_binary)}")


if __name__ == "__main__":
    main()

