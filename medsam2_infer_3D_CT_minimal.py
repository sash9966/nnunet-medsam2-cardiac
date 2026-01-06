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


def run_medsam2_single_label(
    predictor,
    ct_np,            # (D, H, W) pre-windowed and rescaled to 0–255 uint8
    coarse_mask_3d,   # (D, H, W) binary mask from nnUNet, same shape as ct_np
):
    """
    Per-slice 2D MedSAM2 refinement:
    - Find all slices where nnUNet mask has non-zero voxels
    - For each such slice, independently refine the 2D contour using MedSAM2
    - Stack refined slices back into 3D
    - This preserves nnUNet's 3D structure while refining contours slice-by-slice
    Returns: segs_3D (D, H, W) binary mask
    """
    D, H, W = ct_np.shape
    segs_3D = np.zeros((D, H, W), dtype=np.uint8)

    # Find all slices where nnUNet found this structure
    slices_with_mask = []
    for z in range(D):
        if np.any(coarse_mask_3d[z] > 0.5):
            slices_with_mask.append(z)
    
    if len(slices_with_mask) == 0:
        print(f"    Warning: coarse mask is completely empty, skipping")
        return segs_3D
    
    print(f"    Refining {len(slices_with_mask)} slices (z={slices_with_mask[0]}..{slices_with_mask[-1]})")

    # Prepare video tensor (all slices at once)
    video_height, video_width = H, W
    img_resized = resize_grayscale_to_rgb_and_resize(ct_np, 512)  # (D, 3, 512, 512)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = torch.from_numpy(img_resized).to(predictor.device)

    # Normalize as in other scripts
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None].to(predictor.device)
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None].to(predictor.device)
    img_resized = (img_resized - img_mean) / img_std

    # Initialize predictor state once for all slices
    with torch.inference_mode():
        inference_state = predictor.init_state(
            images=img_resized,
            video_height=video_height,
            video_width=video_width,
        )

        # Refine each slice independently
        # Reset state between slices to ensure no cross-contamination
        for z in slices_with_mask:
            # Extract 2D coarse mask for this slice (boolean, original resolution)
            coarse_2d = coarse_mask_3d[z] > 0.5
            
            if np.sum(coarse_2d) == 0:
                continue  # Skip if somehow empty (shouldn't happen, but safety check)

            # Reset state before each slice to ensure independence
            predictor.reset_state(inference_state)
            
            # Re-initialize state for this slice (needed after reset)
            inference_state = predictor.init_state(
                images=img_resized,
                video_height=video_height,
                video_width=video_width,
            )

            # Use add_new_mask to refine this specific slice
            # Note: add_new_mask returns masks at original video resolution
            frame_idx, obj_ids, masks = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=z,
                obj_id=1,
                mask=coarse_2d,
            )

            # Extract refined mask: masks is [num_objects, H, W] at original video resolution
            mask_tensor = masks[0]  # Get first (and only) object's mask
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.squeeze(0)  # Remove channel dim if present
            
            # Convert to boolean numpy array at original resolution
            mask_refined_2d = (mask_tensor > 0.0).cpu().numpy()
            
            # Ensure it's 2D boolean
            if mask_refined_2d.ndim != 2:
                print(f"    Warning: Unexpected mask shape {mask_refined_2d.shape} for slice {z}")
                continue
            
            # Apply refined mask to this slice in the 3D volume
            segs_3D[z][mask_refined_2d] = 1

        # Final reset
        predictor.reset_state(inference_state)

    return segs_3D


def main():
    parser = argparse.ArgumentParser(
        description="Minimal MedSAM2 inference: refine nnUNet prompts on key slice only."
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
        help='Directory containing JSON prompt files (from nnunet_to_medsam2_prompts.py)',
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
    print(f"Using device: {device} for MedSAM2 inference")

    predictor = build_sam2_video_predictor_npz(
        config_file=model_cfg,
        ckpt_path=checkpoint,
        device=device,
    )

    # Find JSON prompt files
    json_files = sorted(
        f for f in os.listdir(prompts_dir)
        if f.endswith('.json') and not f.startswith('._')
    )
    print(f"Found {len(json_files)} prompt JSON files in {prompts_dir}")

    for json_file in tqdm(json_files, desc="Cases"):
        json_path = join(prompts_dir, json_file)
        with open(json_path, 'r') as f:
            case_data = json.load(f)

        case_id = case_data.get('case_id', json_file.replace('.json', ''))
        image_path = case_data['image_path']

        if not exists(image_path):
            print(f"[WARN] CT image not found for {case_id}: {image_path}")
            continue

        print(f"\nProcessing case {case_id}")

        # Load CT as numpy (D, H, W)
        ct_img = sitk.ReadImage(image_path)
        ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)

        # Simple CT windowing (soft tissue-ish)
        lower_bound, upper_bound = -160.0, 240.0
        ct_np = np.clip(ct_np, lower_bound, upper_bound)
        ct_np = (ct_np - ct_np.min()) / (ct_np.max() - ct_np.min() + 1e-8) * 255.0
        ct_np = ct_np.astype(np.uint8)

        prompts = case_data.get('prompts', [])
        if not prompts:
            print(f"  [WARN] No prompts found for {case_id}")
            continue

        for prompt in prompts:
            label_name = prompt.get('label_name', f"label_{prompt.get('label_id', 0)}")
            centroid = prompt['centroid_voxel']  # [z, y, x]
            coarse_mask_path = prompt['coarse_mask_path']

            print(f"  Label {label_name}: centroid={centroid}, coarse_mask={coarse_mask_path}")

            if not exists(coarse_mask_path):
                print(f"    [WARN] Coarse mask not found: {coarse_mask_path}")
                continue

            # Load 3D coarse mask
            coarse_img = sitk.ReadImage(coarse_mask_path)
            coarse_np = sitk.GetArrayFromImage(coarse_img).astype(np.uint8)

            # Sanity: ensure coarse mask is not empty
            if np.count_nonzero(coarse_np) == 0:
                print(f"    [WARN] Coarse mask is completely empty for {label_name}, skipping")
                continue

            # Check shape alignment
            if coarse_np.shape != ct_np.shape:
                print(f"    [WARN] Shape mismatch CT {ct_np.shape} vs coarse {coarse_np.shape}, skipping")
                continue

            # Refine all slices where nnUNet found this structure
            # The function will find all slices with non-zero voxels and refine each independently
            segs_3D = run_medsam2_single_label(
                predictor=predictor,
                ct_np=ct_np,
                coarse_mask_3d=coarse_np,
            )

            # Save refined mask as binary (0/1) uint8
            # Ensure values are exactly 0 or 1
            mask_binary = (segs_3D > 0).astype(np.uint8)
            out_mask_img = sitk.GetImageFromArray(mask_binary)
            out_mask_img.CopyInformation(ct_img)

            save_name = f"{case_id}_{label_name}_mask.nii.gz"
            out_path = join(output_dir, save_name)
            sitk.WriteImage(out_mask_img, out_path)
            
            # Verify what we saved
            num_voxels = np.count_nonzero(mask_binary)
            print(f"    Saved refined mask: {out_path}")
            print(f"      Non-zero voxels: {num_voxels}, Unique values: {np.unique(mask_binary)}")


if __name__ == "__main__":
    main()


