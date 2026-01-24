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
import torch.multiprocessing as mp
from scipy.ndimage import zoom
# Import from MedSAM2 - adjust path if needed
import sys
sys.path.insert(0, 'MedSAM2')
import sam2
from sam2.build_sam import build_sam2_video_predictor_npz
import SimpleITK as sitk
from skimage import measure, morphology

# Detect available device: use CUDA if available, otherwise CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2024)
np.random.seed(2024)

# Structure classification
HIGH_CONTRAST_STRUCTURES = ["LV", "RV", "Aorta", "Pulmonary"]
TOPOLOGICALLY_COMPLEX_STRUCTURES = ["Myo"]  # Myocardium and VSD areas

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="checkpoints/MedSAM2_latest.pt",
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="configs/sam2.1_hiera_t512.yaml",
    help='model config',
)

parser.add_argument(
    '-i',
    '--imgs_path',
    type=str,
    default="CT_DeepLesion/images",
    help='imgs path',
)
parser.add_argument(
    '--gts_path',
    default=None,
    help='simulate prompts based on ground truth',
)
parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="./medsam2_results_uncertainty",
    help='path to save segmentation results',
)
parser.add_argument(
    '--prompts_dir',
    type=str,
    default=None,
    help='directory containing JSON prompt files (e.g., prompts_out/). If not specified, will search in pred_save_dir and other locations',
)
parser.add_argument(
    '--nnunet_seg_dir',
    type=str,
    default=None,
    help='directory containing nnU-Net segmentation files (for fallback and ensemble)',
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
imgs_path = args.imgs_path
gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
prompts_dir = args.prompts_dir
nnunet_seg_dir = args.nnunet_seg_dir
os.makedirs(pred_save_dir, exist_ok=True)

# Handle config path - construct absolute path for Hydra
# Hydra expects either a config name it can resolve or an absolute path with // prefix
script_directory = os.path.dirname(os.path.abspath(__file__))
medsam2_dir = os.path.join(script_directory, 'MedSAM2')

# Normalize the config path
if model_cfg.startswith('MedSAM2/'):
    # Remove MedSAM2/ prefix
    model_cfg = model_cfg.replace('MedSAM2/', '')
elif model_cfg.startswith('sam2/'):
    # Keep as is
    pass
elif not model_cfg.startswith('configs/'):
    # Assume it's a config name, try to find it
    if os.path.exists(os.path.join(medsam2_dir, 'sam2', 'configs', model_cfg)):
        model_cfg = os.path.join('sam2', 'configs', model_cfg)
    elif os.path.exists(os.path.join(medsam2_dir, 'sam2', 'configs', os.path.basename(model_cfg))):
        model_cfg = os.path.join('sam2', 'configs', os.path.basename(model_cfg))

# Construct absolute path with // prefix for Hydra (similar to medsam2_infer_CT_lesion_npz_recist.py)
config_full_path = os.path.join(medsam2_dir, model_cfg)
if os.path.exists(config_full_path):
    model_cfg = '//' + config_full_path
else:
    print(f"Warning: Config file not found at {config_full_path}, trying default resolution")

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def dice_multi_class(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]
    dices = []
    for label in labels:
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices.append((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    return np.mean(dices)

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def mask2D_to_bbox(gt2D, max_shift=20):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D, max_shift=20):
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    D, H, W = gt3D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    z_min = max(0, z_min)
    z_max = min(D-1, z_max)
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes3d

def load_sitk_mask(mask_path):
    """Load a mask file using SimpleITK and return as numpy array."""
    mask_img = sitk.ReadImage(mask_path)
    mask_data = sitk.GetArrayFromImage(mask_img)
    return mask_data, mask_img

def load_nnunet_segmentation(nnunet_seg_dir, case_id):
    """Load nnU-Net segmentation for a case."""
    if nnunet_seg_dir is None:
        return None
    
    seg_path = os.path.join(nnunet_seg_dir, f"{case_id}.nii.gz")
    if not exists(seg_path):
        return None
    
    try:
        seg_img = sitk.ReadImage(seg_path)
        seg_data = sitk.GetArrayFromImage(seg_img)
        return seg_data, seg_img
    except Exception as e:
        print(f"  Warning: Could not load nnU-Net segmentation: {e}")
        return None

# Initialize predictor
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint, device=device)

# Find JSON files
json_files = []
json_search_paths = []
if prompts_dir:
    json_search_paths.append(prompts_dir)
json_search_paths.extend([
    pred_save_dir,
    join(os.path.dirname(imgs_path), 'prompts_out'),
    join(os.path.dirname(os.path.abspath(imgs_path)), 'prompts_out'),
    'prompts_out',
])

for search_path in json_search_paths:
    if search_path and exists(search_path):
        json_files.extend([f for f in os.listdir(search_path) if f.endswith('.json') and not f.startswith('._')])
        if json_files:
            print(f'Found {len(json_files)} JSON files in {search_path}')
            break

if not json_files:
    print(f'No JSON files found. Searched in: {json_search_paths}')
    print('Exiting - uncertainty-guided inference requires JSON prompt files with uncertainty analysis')
    exit(1)

# Load JSON files and extract image paths
cases_to_process = []
json_search_dir = search_path if json_files else None

for json_file in sorted(json_files):
    json_path = join(json_search_dir, json_file) if json_search_dir else json_file
    try:
        with open(json_path, 'r') as f:
            case_data = json.load(f)
            image_path = case_data.get('image_path')
            case_id = case_data.get('case_id', json_file.replace('.json', ''))
            
            if image_path:
                # Resolve relative paths
                if not os.path.isabs(image_path):
                    json_dir = os.path.dirname(os.path.abspath(json_path))
                    abs_image_path = join(json_dir, image_path)
                    if not exists(abs_image_path):
                        abs_image_path = image_path
                        if not exists(abs_image_path):
                            abs_image_path = join(imgs_path, os.path.basename(image_path))
                    
                    if exists(abs_image_path):
                        image_path = abs_image_path
                    else:
                        print(f"Warning: Image path {image_path} not found for {case_id}, skipping")
                        continue
                
                if exists(image_path):
                    cases_to_process.append((image_path, case_id, json_path, case_data))
                    print(f"  Found case {case_id}: {image_path}")
                else:
                    print(f"Warning: Image file not found: {image_path} for case {case_id}")
            else:
                print(f"Warning: No image_path in JSON file {json_file}")
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        continue

print(f'Processing {len(cases_to_process)} cases from JSON files')

seg_info = OrderedDict()
seg_info['nii_name'] = []
seg_info['key_slice_index'] = []
seg_info['DICOM_windows'] = []

# Process cases
for case_info in tqdm(cases_to_process):
    image_path, case_identifier, json_path, json_data = case_info
    case_id = case_identifier
    
    # Load the CT image
    if not exists(image_path):
        print(f"Skipping {case_identifier}: image file not found: {image_path}")
        continue
    
    nii_image = sitk.ReadImage(image_path)
    nii_image_data = sitk.GetArrayFromImage(nii_image)
    nii_fname = os.path.basename(image_path)
    
    # Store original image data for saving
    nii_image_data_original = nii_image_data.copy()
    
    # Load nnU-Net segmentation if available
    nnunet_seg_data, nnunet_seg_img = load_nnunet_segmentation(nnunet_seg_dir, case_id) or (None, None)
    
    case_data = json_data
    prompts_list = enumerate(case_data.get('prompts', []))
    
    if not case_data.get('prompts'):
        print(f"  Warning: No prompts found in JSON for {case_id}")
        continue

    for row_id, prompt_data in prompts_list:
        # Initialize segmentation for this prompt
        segs_3D = np.zeros(nii_image_data.shape, dtype=np.uint8)
        
        # Extract information from JSON
        voxel_bbox = prompt_data['voxel_bbox']  # [zmin, ymin, xmin, zmax, ymax, xmax]
        centroid = prompt_data['centroid_voxel']  # [z, y, x]
        label_name = prompt_data.get('label_name', f"label_{prompt_data.get('label_id', row_id)}")
        label_id = prompt_data.get('label_id', row_id + 1)
        coarse_mask_path = prompt_data.get('coarse_mask_path', None)
        
        # Get uncertainty analysis data
        uncertainty_analysis = prompt_data.get('uncertainty_analysis', {})
        high_conf_seeds = uncertainty_analysis.get('high_confidence_seeds', [])
        is_high_contrast = uncertainty_analysis.get('is_high_contrast', False)
        is_topologically_complex = uncertainty_analysis.get('is_topologically_complex', False)
        has_vsd = uncertainty_analysis.get('has_vsd', False)
        
        print(f"  Processing {label_name} (label_id={label_id})")
        print(f"    High-contrast: {is_high_contrast}, Topologically complex: {is_topologically_complex}, VSD: {has_vsd}")
        
        # CRUCIAL: Skip MedSAM refinement for topologically complex structures
        if is_topologically_complex or has_vsd:
            print(f"    Skipping MedSAM refinement for {label_name} (topologically complex/VSD)")
            # Use nnU-Net segmentation directly
            if nnunet_seg_data is not None:
                label_mask = (nnunet_seg_data == label_id).astype(np.uint8)
                segs_3D = label_mask
                print(f"    Using nnU-Net segmentation directly ({np.sum(segs_3D > 0)} voxels)")
            else:
                print(f"    Warning: No nnU-Net segmentation available, skipping {label_name}")
                continue
        else:
            # Use MedSAM2 refinement for high-contrast structures
            print(f"    Using MedSAM2 refinement for {label_name}")
            
            # Use default DICOM window (soft tissue window)
            lower_bound, upper_bound = -160.0, 240.0
            
            # Key slice selection using Weighted Centroid approach
            # Load nnU-Net mask to determine optimal key slice
            if nnunet_seg_data is not None:
                # Get mask for this specific label
                nnunet_label_mask = (nnunet_seg_data == label_id).astype(bool)
                
                # Find z-coordinates where mask is active
                z_coords = np.where(nnunet_label_mask)[0]
                if len(z_coords) > 0:
                    z_min, z_max = z_coords.min(), z_coords.max()
                    z_mid = (z_min + z_max) // 2
                    
                    # Find the slice with the most pixels, but only within the middle 40% of the mask
                    buffer = int((z_max - z_min) * 0.2)
                    search_range = range(max(0, z_mid - buffer), min(nnunet_label_mask.shape[0], z_mid + buffer + 1))
                    
                    best_slice = z_mid
                    max_area = 0
                    for z in search_range:
                        if z < nnunet_label_mask.shape[0]:
                            area = np.sum(nnunet_label_mask[z] > 0)
                            if area > max_area:
                                max_area = area
                                best_slice = z
                    
                    key_slice_idx = int(best_slice)
                    print(f"    Using central key slice: {key_slice_idx} (area: {max_area} pixels, z-range: {z_min}-{z_max})")
                else:
                    # Fallback to centroid if mask is empty
                    key_slice_idx = int(round(centroid[0]))
                    print(f"    Warning: Empty mask, using centroid slice: {key_slice_idx}")
            else:
                # Fallback to centroid if nnU-Net segmentation not available
                key_slice_idx = int(round(centroid[0]))
                print(f"    Warning: nnU-Net segmentation not available, using centroid slice: {key_slice_idx}")
            
            # Ensure key slice is within bounds
            key_slice_idx = max(0, min(key_slice_idx, nii_image_data.shape[0] - 1))
            key_slice_idx_offset = key_slice_idx
            
            if key_slice_idx_offset < 0 or key_slice_idx_offset >= nii_image_data.shape[0]:
                print(f"    Warning: key_slice_idx_offset {key_slice_idx_offset} out of bounds, using slice 0")
                key_slice_idx_offset = 0
                key_slice_idx = 0
            
            # Preprocess image with DICOM window
            nii_image_data_pre = np.clip(nii_image_data, lower_bound, upper_bound)
            nii_image_data_pre = (nii_image_data_pre - np.min(nii_image_data_pre))/(np.max(nii_image_data_pre)-np.min(nii_image_data_pre))*255.0
            nii_image_data_pre = np.uint8(nii_image_data_pre)
            
            key_slice_img = nii_image_data_pre[key_slice_idx_offset, :,:]
            
            img_3D_ori = nii_image_data_pre
            assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'
            
            video_height = key_slice_img.shape[0]
            video_width = key_slice_img.shape[1]
            img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
            img_resized = img_resized / 255.0
            img_resized = torch.from_numpy(img_resized).to(device)
            img_mean=(0.485, 0.456, 0.406)
            img_std=(0.229, 0.224, 0.225)
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].to(device)
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].to(device)
            img_resized -= img_mean
            img_resized /= img_std
            
            # Load coarse mask if available
            coarse_mask_2d = None
            if coarse_mask_path:
                if not os.path.isabs(coarse_mask_path):
                    for base_dir in [prompts_dir, pred_save_dir, json_search_dir]:
                        if base_dir:
                            test_path = join(base_dir, os.path.basename(coarse_mask_path))
                            if exists(test_path):
                                coarse_mask_path = test_path
                                break
                            test_path = join(base_dir, coarse_mask_path.replace('prompts_out/', ''))
                            if exists(test_path):
                                coarse_mask_path = test_path
                                break
                
                if exists(coarse_mask_path):
                    try:
                        coarse_mask_3d, _ = load_sitk_mask(coarse_mask_path)
                        if key_slice_idx_offset < coarse_mask_3d.shape[0]:
                            coarse_mask_2d = coarse_mask_3d[key_slice_idx_offset, :, :] > 0.5
                            print(f"    Using coarse mask from nnUNet for initialization")
                    except Exception as e:
                        print(f"    Warning: Could not load coarse mask: {e}")
            
            # Prepare point prompts from high-confidence seeds
            points_2d = None
            labels_2d = None
            
            if len(high_conf_seeds) > 0:
                # Filter seeds to the key slice (within ±2 slices)
                key_slice_seeds = []
                for seed in high_conf_seeds:
                    z, y, x = seed
                    if abs(z - key_slice_idx_offset) <= 2:
                        # Project to key slice if needed, or use directly if on key slice
                        if z == key_slice_idx_offset:
                            key_slice_seeds.append([x, y])  # Note: SAM2 expects [x, y] format
                        else:
                            # Use seeds from nearby slices
                            key_slice_seeds.append([x, y])
                
                if len(key_slice_seeds) > 0:
                    # Sample up to 10 points
                    n_points = min(10, len(key_slice_seeds))
                    if len(key_slice_seeds) > n_points:
                        indices = np.random.choice(len(key_slice_seeds), size=n_points, replace=False)
                        key_slice_seeds = [key_slice_seeds[i] for i in indices]
                    
                    points_2d = np.array(key_slice_seeds, dtype=np.float32)
                    labels_2d = np.ones(len(key_slice_seeds), dtype=np.int32)  # All positive points
                    print(f"    Using {len(points_2d)} point prompts from high-confidence seeds")
            
            # Fallback to bounding box if no points available
            if points_2d is None or len(points_2d) == 0:
                H, W = nii_image_data.shape[1], nii_image_data.shape[2]
                x_min = max(0, min(int(voxel_bbox[2]), W - 1))
                y_min = max(0, min(int(voxel_bbox[1]), H - 1))
                x_max = max(x_min + 1, min(int(voxel_bbox[5]), W - 1))
                y_max = max(y_min + 1, min(int(voxel_bbox[4]), H - 1))
                bbox = np.array([x_min, y_min, x_max, y_max])
                print(f"    Falling back to bounding box: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            # Use autocast for CUDA (mixed precision), regular inference for CPU
            if device.type == "cuda":
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inference_state = predictor.init_state(img_resized, video_height, video_width)
                    
                    # Initialize with points or coarse mask or bbox
                    if coarse_mask_2d is not None:
                        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            mask=coarse_mask_2d,
                        )
                        print(f"    Initialized with coarse mask on slice {key_slice_idx_offset}")
                    elif points_2d is not None and len(points_2d) > 0:
                        # Convert points to torch tensor
                        points_tensor = torch.from_numpy(points_2d).to(device).unsqueeze(0)  # [1, N, 2]
                        labels_tensor = torch.from_numpy(labels_2d).to(device).unsqueeze(0)  # [1, N]
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            points=points_tensor,
                            labels=labels_tensor,
                            box=None,
                        )
                        print(f"    Initialized with {len(points_2d)} point prompts on slice {key_slice_idx_offset}")
                    else:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            box=bbox,
                        )
                        print(f"    Initialized with bounding box on slice {key_slice_idx_offset}")
                    
                    # Forward propagation
                    # Follow exact pattern from medsam2_infer_CT_lesion_npz_recist.py
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                        # Follow exact pattern from medsam2_infer_CT_lesion_npz_recist.py line 383
                        # out_mask_logits[0] shape is [1, H, W], need [0] after numpy conversion
                        mask_2d = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                        
                        # Debug: print shape on first few frames
                        if out_frame_idx <= key_slice_idx_offset + 2:
                            print(f"    Frame {out_frame_idx}: out_mask_logits[0] shape: {out_mask_logits[0].shape}, mask_2d shape: {mask_2d.shape}, dtype: {mask_2d.dtype}")
                            print(f"    Frame {out_frame_idx}: mask_2d non-zero: {np.sum(mask_2d > 0)}")
                            print(f"    Frame {out_frame_idx}: segs_3D shape: {segs_3D.shape}, expected mask shape: {(segs_3D.shape[1], segs_3D.shape[2])}")
                        
                        # Ensure mask_2d is 2D [H, W] boolean
                        if mask_2d.ndim != 2:
                            print(f"    Warning: mask_2d has {mask_2d.ndim} dimensions, expected 2. Shape: {mask_2d.shape}")
                            continue
                        
                        mask_2d = mask_2d.astype(bool)
                        
                        # Verify mask dimensions match segs_3D slice dimensions
                        expected_shape = (segs_3D.shape[1], segs_3D.shape[2])
                        if mask_2d.shape != expected_shape:
                            # Resize mask to match segs_3D dimensions if needed
                            zoom_factors = (expected_shape[0] / mask_2d.shape[0], expected_shape[1] / mask_2d.shape[1])
                            mask_2d = zoom(mask_2d.astype(float), zoom_factors, order=0) > 0.5
                            if out_frame_idx <= key_slice_idx_offset + 2:
                                print(f"    Frame {out_frame_idx}: Resized mask from {mask_2d.shape} to {expected_shape}")
                        
                        # Apply mask to segmentation - use exact pattern from working example
                        # segs_3D shape: [D, H, W], mask_2d shape: [H, W]
                        if out_frame_idx < segs_3D.shape[0]:
                            # Use boolean indexing: segs_3D[z, y, x] where mask_2d[y, x] is True
                            segs_3D[out_frame_idx, mask_2d] = 1
                            
                            # Debug: check if any voxels were set
                            if out_frame_idx <= key_slice_idx_offset + 2:
                                num_set = np.sum(segs_3D[out_frame_idx] > 0)
                                print(f"    Frame {out_frame_idx}: Set {num_set} voxels (mask had {np.sum(mask_2d)} True values)")
                        else:
                            print(f"    Warning: out_frame_idx {out_frame_idx} >= segs_3D.shape[0] {segs_3D.shape[0]}")
                    
                    predictor.reset_state(inference_state)
                    # Re-initialize inference state for reverse propagation (CRITICAL: must re-init after reset)
                    inference_state = predictor.init_state(img_resized, video_height, video_width)
                    
                    # Re-initialize for reverse propagation
                    if coarse_mask_2d is not None:
                        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            mask=coarse_mask_2d,
                        )
                    elif points_2d is not None and len(points_2d) > 0:
                        points_tensor = torch.from_numpy(points_2d).to(device).unsqueeze(0)
                        labels_tensor = torch.from_numpy(labels_2d).to(device).unsqueeze(0)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            points=points_tensor,
                            labels=labels_tensor,
                            box=None,
                        )
                    else:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            box=bbox,
                        )
                    
                    # Reverse propagation
                    # Follow exact pattern from medsam2_infer_CT_lesion_npz_recist.py
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                        # out_mask_logits shape: [num_objects, H, W] or [num_objects, 1, H, W]
                        # Get first object's mask
                        mask_tensor = out_mask_logits[0]  # Shape: [H, W] or [1, H, W]
                        mask_2d = (mask_tensor > 0.0).cpu().numpy()
                        # Squeeze to remove any singleton dimensions
                        if mask_2d.ndim == 3:
                            mask_2d = mask_2d.squeeze(0)  # Remove channel dimension if present
                        elif mask_2d.ndim == 1:
                            # If somehow 1D, skip this frame
                            print(f"    Warning: Unexpected mask shape {mask_2d.shape} for frame {out_frame_idx}")
                            continue
                        
                        # Ensure mask_2d is boolean
                        mask_2d = mask_2d.astype(bool)
                        
                        # Verify mask dimensions match segs_3D slice dimensions
                        expected_shape = (segs_3D.shape[1], segs_3D.shape[2])
                        if mask_2d.shape != expected_shape:
                            # Resize mask to match segs_3D dimensions if needed
                            zoom_factors = (expected_shape[0] / mask_2d.shape[0], expected_shape[1] / mask_2d.shape[1])
                            mask_2d = zoom(mask_2d.astype(float), zoom_factors, order=0) > 0.5
                        
                        # Apply mask to segmentation (mask_2d should be [H, W] boolean)
                        segs_3D[out_frame_idx, mask_2d] = 1
                    
                    predictor.reset_state(inference_state)
                    # Re-initialize inference state for reverse propagation (CRITICAL: must re-init after reset)
                    inference_state = predictor.init_state(img_resized, video_height, video_width)
                    
                    # Re-initialize for reverse propagation
                    if coarse_mask_2d is not None:
                        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            mask=coarse_mask_2d,
                        )
                    elif points_2d is not None and len(points_2d) > 0:
                        points_tensor = torch.from_numpy(points_2d).to(device).unsqueeze(0)
                        labels_tensor = torch.from_numpy(labels_2d).to(device).unsqueeze(0)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            points=points_tensor,
                            labels=labels_tensor,
                            box=None,
                        )
                    else:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            box=bbox,
                        )
                    
                    # Reverse propagation
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                        mask_tensor = out_mask_logits[0]
                        mask_2d = (mask_tensor > 0.0).cpu().numpy()
                        if mask_2d.ndim == 3:
                            mask_2d = mask_2d.squeeze(0)
                        elif mask_2d.ndim == 1:
                            print(f"    Warning: Unexpected mask shape {mask_2d.shape} for frame {out_frame_idx}")
                            continue
                        mask_2d = mask_2d.astype(bool)
                        expected_shape = (segs_3D.shape[1], segs_3D.shape[2])
                        if mask_2d.shape != expected_shape:
                            zoom_factors = (expected_shape[0] / mask_2d.shape[0], expected_shape[1] / mask_2d.shape[1])
                            mask_2d = zoom(mask_2d.astype(float), zoom_factors, order=0) > 0.5
                        segs_3D[out_frame_idx, mask_2d] = 1
                    
                    predictor.reset_state(inference_state)
            else:
                # CPU inference without autocast
                with torch.inference_mode():
                    inference_state = predictor.init_state(img_resized, video_height, video_width)
                    
                    if coarse_mask_2d is not None:
                        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            mask=coarse_mask_2d,
                        )
                    elif points_2d is not None and len(points_2d) > 0:
                        points_tensor = torch.from_numpy(points_2d).to(device).unsqueeze(0)
                        labels_tensor = torch.from_numpy(labels_2d).to(device).unsqueeze(0)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            points=points_tensor,
                            labels=labels_tensor,
                            box=None,
                        )
                    else:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            box=bbox,
                        )
                    
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                        # Follow exact pattern from medsam2_infer_CT_lesion_npz_recist.py line 383
                        # out_mask_logits[0] shape is [1, H, W], need [0] after numpy conversion
                        mask_2d = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                        
                        # Debug: print shape on first few frames
                        if out_frame_idx <= key_slice_idx_offset + 2:
                            print(f"    Frame {out_frame_idx}: out_mask_logits[0] shape: {out_mask_logits[0].shape}, mask_2d shape: {mask_2d.shape}, dtype: {mask_2d.dtype}")
                            print(f"    Frame {out_frame_idx}: mask_2d non-zero: {np.sum(mask_2d > 0)}")
                            print(f"    Frame {out_frame_idx}: segs_3D shape: {segs_3D.shape}, expected mask shape: {(segs_3D.shape[1], segs_3D.shape[2])}")
                        
                        # Ensure mask_2d is 2D [H, W] boolean
                        if mask_2d.ndim != 2:
                            print(f"    Warning: mask_2d has {mask_2d.ndim} dimensions, expected 2. Shape: {mask_2d.shape}")
                            continue
                        
                        mask_2d = mask_2d.astype(bool)
                        
                        # Verify mask dimensions match segs_3D slice dimensions
                        expected_shape = (segs_3D.shape[1], segs_3D.shape[2])
                        if mask_2d.shape != expected_shape:
                            # Resize mask to match segs_3D dimensions if needed
                            zoom_factors = (expected_shape[0] / mask_2d.shape[0], expected_shape[1] / mask_2d.shape[1])
                            mask_2d = zoom(mask_2d.astype(float), zoom_factors, order=0) > 0.5
                            if out_frame_idx <= key_slice_idx_offset + 2:
                                print(f"    Frame {out_frame_idx}: Resized mask from {mask_2d.shape} to {expected_shape}")
                        
                        # Apply mask to segmentation - use exact pattern from working example
                        # segs_3D shape: [D, H, W], mask_2d shape: [H, W]
                        if out_frame_idx < segs_3D.shape[0]:
                            # Use boolean indexing: segs_3D[z, y, x] where mask_2d[y, x] is True
                            segs_3D[out_frame_idx, mask_2d] = 1
                            
                            # Debug: check if any voxels were set
                            if out_frame_idx <= key_slice_idx_offset + 2:
                                num_set = np.sum(segs_3D[out_frame_idx] > 0)
                                print(f"    Frame {out_frame_idx}: Set {num_set} voxels (mask had {np.sum(mask_2d)} True values)")
                        else:
                            print(f"    Warning: out_frame_idx {out_frame_idx} >= segs_3D.shape[0] {segs_3D.shape[0]}")
                    
                    predictor.reset_state(inference_state)
                    
                    if coarse_mask_2d is not None:
                        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            mask=coarse_mask_2d,
                        )
                    elif points_2d is not None and len(points_2d) > 0:
                        points_tensor = torch.from_numpy(points_2d).to(device).unsqueeze(0)
                        labels_tensor = torch.from_numpy(labels_2d).to(device).unsqueeze(0)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            points=points_tensor,
                            labels=labels_tensor,
                            box=None,
                        )
                    else:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=key_slice_idx_offset,
                            obj_id=1,
                            box=bbox,
                        )
                    
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                        # out_mask_logits shape: [num_objects, H, W] or [num_objects, 1, H, W]
                        # Get first object's mask
                        mask_tensor = out_mask_logits[0]  # Shape: [H, W] or [1, H, W]
                        mask_2d = (mask_tensor > 0.0).cpu().numpy()
                        # Squeeze to remove any singleton dimensions
                        if mask_2d.ndim == 3:
                            mask_2d = mask_2d.squeeze(0)  # Remove channel dimension if present
                        elif mask_2d.ndim == 1:
                            # If somehow 1D, skip this frame
                            print(f"    Warning: Unexpected mask shape {mask_2d.shape} for frame {out_frame_idx}")
                            continue
                        
                        # Ensure mask_2d is boolean
                        mask_2d = mask_2d.astype(bool)
                        
                        # Verify mask dimensions match segs_3D slice dimensions
                        expected_shape = (segs_3D.shape[1], segs_3D.shape[2])
                        if mask_2d.shape != expected_shape:
                            # Resize mask to match segs_3D dimensions if needed
                            zoom_factors = (expected_shape[0] / mask_2d.shape[0], expected_shape[1] / mask_2d.shape[1])
                            mask_2d = zoom(mask_2d.astype(float), zoom_factors, order=0) > 0.5
                        
                        # Apply mask to segmentation (mask_2d should be [H, W] boolean)
                        segs_3D[out_frame_idx, mask_2d] = 1
                    
                    predictor.reset_state(inference_state)
        
        # Post-process segmentation
        if np.max(segs_3D) > 0:
            segs_3D = getLargestCC(segs_3D)
            segs_3D = np.uint8(segs_3D)
        else:
            print(f"    Warning: Empty segmentation for {label_name}")
        
        # Save segmentation mask
        # Follow exact pattern from medsam2_infer_3D_CT_modified.py
        sitk_mask = sitk.GetImageFromArray(segs_3D.astype(np.uint8))
        sitk_mask.CopyInformation(nii_image)
        
        # Save segmentation
        save_seg_name = nii_fname.split('.nii.gz')[0] + f'_{label_name}_mask.nii.gz'
        dicom_windows_str = f"{lower_bound}, {upper_bound}"
        
        # Save original CT image once per case
        if row_id == 0:
            sitk_image_original = sitk.GetImageFromArray(nii_image_data_original)
            sitk_image_original.CopyInformation(nii_image)
            sitk.WriteImage(sitk_image_original, os.path.join(pred_save_dir, nii_fname.replace('.nii.gz', '_img.nii.gz')))
            print(f"  Saved original CT image: {nii_fname.replace('.nii.gz', '_img.nii.gz')}")
        
        # Save mask
        sitk.WriteImage(sitk_mask, os.path.join(pred_save_dir, save_seg_name))
        print(f"  Saved {label_name} mask: {save_seg_name} (shape: {segs_3D.shape}, non-zero voxels: {np.sum(segs_3D > 0)})")
        seg_info['nii_name'].append(save_seg_name)
        seg_info['key_slice_index'].append(key_slice_idx)
        seg_info['DICOM_windows'].append(dicom_windows_str)

# Save segmentation info after processing all cases
if seg_info['nii_name']:
    seg_info_df = pd.DataFrame(seg_info)
    seg_info_df.to_csv(join(pred_save_dir, 'tiny_seg_info202412.csv'), index=False)
    print(f"Saved segmentation info to {join(pred_save_dir, 'tiny_seg_info202412.csv')}")
