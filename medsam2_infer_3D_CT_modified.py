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
# Import from MedSAM2 - adjust path if needed
# If running from root directory with MedSAM2 as subdirectory:
import sys
sys.path.insert(0, 'MedSAM2')
# Import sam2 to initialize Hydra config module
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
    default="./DeeLesion_results",
    help='path to save segmentation results',
)
parser.add_argument(
    '--prompts_dir',
    type=str,
    default=None,
    help='directory containing JSON prompt files (e.g., prompts_out/). If not specified, will search in pred_save_dir and other locations',
)
# add option to propagate with either box or mask
parser.add_argument(
    '--propagate_with_box',
    default=True,
    action='store_true',
    help='whether to propagate with box'
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
imgs_path = args.imgs_path
gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
prompts_dir = args.prompts_dir
os.makedirs(pred_save_dir, exist_ok=True)
propagate_with_box = args.propagate_with_box

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

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
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

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
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


# Check if DeepLesion CSV exists, otherwise use JSON prompts
csv_path = 'CT_DeepLesion/DeepLesion_Dataset_Info.csv'
use_csv = exists(csv_path)
if use_csv:
    DL_info = pd.read_csv(csv_path)
    print("Using DeepLesion CSV format")
else:
    DL_info = None
    print("CSV not found, will use JSON prompt files if available")

# Initialize predictor
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint, device=device)

# Determine processing mode: CSV or JSON
if use_csv:
    # CSV mode: process files from imgs_path directory
    nii_fnames = sorted(os.listdir(imgs_path))
    nii_fnames = [i for i in nii_fnames if i.endswith('.nii.gz')]
    nii_fnames = [i for i in nii_fnames if not i.startswith('._')]
    print(f'Processing {len(nii_fnames)} nii files from {imgs_path}')
    cases_to_process = [(join(imgs_path, fname), fname) for fname in nii_fnames]
else:
    # JSON mode: find all JSON files and use image_path from JSON
    json_files = []
    
    # Try multiple locations for JSON files
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
        print('Falling back to processing files from imgs_path directory')
        nii_fnames = sorted(os.listdir(imgs_path))
        nii_fnames = [i for i in nii_fnames if i.endswith('.nii.gz')]
        nii_fnames = [i for i in nii_fnames if not i.startswith('._')]
        cases_to_process = [(join(imgs_path, fname), fname) for fname in nii_fnames]
    else:
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
                            # Try relative to JSON file location
                            json_dir = os.path.dirname(os.path.abspath(json_path))
                            abs_image_path = join(json_dir, image_path)
                            if not exists(abs_image_path):
                                # Try relative to current working directory
                                abs_image_path = image_path
                                if not exists(abs_image_path):
                                    # Try relative to imgs_path
                                    abs_image_path = join(imgs_path, os.path.basename(image_path))
                            
                            if exists(abs_image_path):
                                image_path = abs_image_path
                            else:
                                print(f"Warning: Image path {image_path} not found for {case_id}, skipping")
                                continue
                        
                        if exists(image_path):
                            # Store image path, case_id, JSON path, and JSON data
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
    if use_csv:
        # CSV mode: case_info is (image_path, nii_fname)
        image_path, nii_fname = case_info
        case_identifier = nii_fname
        json_data = None
        json_path = None
    else:
        # JSON mode: case_info is (image_path, case_id, json_path, json_data)
        image_path, case_identifier, json_path, json_data = case_info
    
    # Load the CT image
    if not exists(image_path):
        print(f"Skipping {case_identifier}: image file not found: {image_path}")
        continue
    
    nii_image = sitk.ReadImage(image_path)
    nii_image_data = sitk.GetArrayFromImage(nii_image)
    nii_fname = os.path.basename(image_path)
    
    # Store original image data for saving (before any preprocessing)
    nii_image_data_original = nii_image_data.copy()
    
    # Determine processing mode: CSV or JSON
    if use_csv:
        # Original DeepLesion CSV format
        range_suffix = re.findall(r'\d{3}-\d{3}', nii_fname)
        if not range_suffix:
            print(f"Skipping {nii_fname}: doesn't match DeepLesion format")
            continue
        range_suffix = range_suffix[0]
        slice_range = range_suffix.split('-')
        slice_range = [str(int(s)) for s in slice_range]
        slice_range = ', '.join(slice_range)
        
        case_name_match = re.findall(r'^(\d{6}_\d{2}_\d{2})', nii_fname)
        if not case_name_match:
            print(f"Skipping {nii_fname}: doesn't match DeepLesion format")
            continue
        case_name = case_name_match[0]
        case_df = DL_info[
            DL_info['File_name'].str.contains(case_name) &
            DL_info['Slice_range'].str.contains(slice_range)
        ].copy()
        
        if case_df.empty:
            print(f"No matching entries in CSV for {nii_fname}")
            continue
        
        prompts_list = case_df.iterrows()
    else:
        # JSON prompt format - we already have the JSON data loaded
        case_id = case_identifier
        
        if json_data is None:
            print(f"Error: No JSON data for case {case_id}")
            continue
        
        case_data = json_data
        
        # Verify the image path matches
        json_image_path = case_data.get('image_path', '')
        if json_image_path:
            # Check if paths match (allowing for relative vs absolute)
            try:
                if exists(json_image_path) and exists(image_path):
                    if not os.path.samefile(image_path, json_image_path):
                        print(f"  Note: JSON image_path ({json_image_path}) differs from loaded image ({image_path})")
            except:
                pass  # Paths might be different representations of same file
        
        prompts_list = enumerate(case_data.get('prompts', []))
        
        if not case_data.get('prompts'):
            print(f"  Warning: No prompts found in JSON for {case_id}")
            continue

    # Store original image data (before preprocessing) for saving
    nii_image_data_original = nii_image_data.copy()
    
    for row_id, prompt_data in prompts_list:
        # Initialize segmentation for this prompt
        segs_3D = np.zeros(nii_image_data.shape, dtype=np.uint8)
        # Extract information based on format
        if use_csv:
            # CSV format (pandas row)
            row = prompt_data
            lower_bound, upper_bound = row['DICOM_windows'].split(',')
            lower_bound, upper_bound = float(lower_bound), float(upper_bound)
            key_slice_idx = int(row['Key_slice_index'])
            slice_range = row['Slice_range']
            slice_idx_start, slice_idx_end = slice_range.split(',')
            slice_idx_start, slice_idx_end = int(slice_idx_start), int(slice_idx_end)
            bbox_coords = row['Bounding_boxes']
            bbox_coords = bbox_coords.split(',')
            bbox_coords = [int(float(coord)) for coord in bbox_coords]
            # bbox format: y_min, x_min, y_max, x_max
            bbox = np.array([bbox_coords[1], bbox_coords[0], bbox_coords[3], bbox_coords[2]])
            key_slice_idx_offset = key_slice_idx - slice_idx_start
            # Ensure key slice offset is within bounds
            if key_slice_idx_offset < 0 or key_slice_idx_offset >= nii_image_data.shape[0]:
                print(f"Warning: key_slice_idx_offset {key_slice_idx_offset} out of bounds for {nii_fname}")
                continue
            label_name = f"lesion_{row_id}"
        else:
            # JSON format
            voxel_bbox = prompt_data['voxel_bbox']  # [zmin, ymin, xmin, zmax, ymax, xmax]
            centroid = prompt_data['centroid_voxel']  # [z, y, x]
            label_name = prompt_data.get('label_name', f"label_{prompt_data.get('label_id', row_id)}")
            label_id = prompt_data.get('label_id', row_id + 1)
            coarse_mask_path = prompt_data.get('coarse_mask_path', None)
            
            print(f"  Processing {label_name} (label_id={label_id})")
            print(f"    3D bbox: {voxel_bbox} [zmin, ymin, xmin, zmax, ymax, xmax]")
            print(f"    Centroid: {centroid} [z, y, x]")
            if coarse_mask_path:
                print(f"    Coarse mask available: {coarse_mask_path}")
            
            # Use default DICOM window (soft tissue window) if not specified
            # Common CT window: level=40, width=400
            lower_bound, upper_bound = -160.0, 240.0  # level ± width/2
            
            # Key slice is the z-coordinate of the centroid
            key_slice_idx = int(round(centroid[0]))
            key_slice_idx = max(0, min(key_slice_idx, nii_image_data.shape[0] - 1))
            key_slice_idx_offset = key_slice_idx
            
            # Ensure key slice is within bounds
            if key_slice_idx_offset < 0 or key_slice_idx_offset >= nii_image_data.shape[0]:
                print(f"    Warning: key_slice_idx_offset {key_slice_idx_offset} out of bounds, using slice 0")
                key_slice_idx_offset = 0
                key_slice_idx = 0
            
            print(f"    Using key slice: {key_slice_idx} (offset: {key_slice_idx_offset})")
            
            # Extract 2D bbox from 3D bbox for the key slice
            # bbox format for SAM2: [x_min, y_min, x_max, y_max]
            H, W = nii_image_data.shape[1], nii_image_data.shape[2]
            x_min = max(0, min(int(voxel_bbox[2]), W - 1))
            y_min = max(0, min(int(voxel_bbox[1]), H - 1))
            x_max = max(x_min + 1, min(int(voxel_bbox[5]), W - 1))
            y_max = max(y_min + 1, min(int(voxel_bbox[4]), H - 1))
            bbox = np.array([x_min, y_min, x_max, y_max])
            
            print(f"    2D bbox for slice {key_slice_idx}: [{x_min}, {y_min}, {x_max}, {y_max}] [x_min, y_min, x_max, y_max]")
        
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
        z_mids = []

        # Load coarse mask if available (for better initialization)
        coarse_mask_2d = None
        if not use_csv and 'coarse_mask_path' in prompt_data and prompt_data.get('coarse_mask_path'):
            coarse_mask_path = prompt_data['coarse_mask_path']
            # Resolve path relative to JSON file location or current directory
            if not os.path.isabs(coarse_mask_path):
                # Try relative to prompts_dir or pred_save_dir
                for base_dir in [prompts_dir, pred_save_dir]:
                    if base_dir:
                        test_path = join(base_dir, os.path.basename(coarse_mask_path))
                        if exists(test_path):
                            coarse_mask_path = test_path
                            break
                        # Also try with full relative path
                        test_path = join(base_dir, coarse_mask_path.replace('prompts_out/', ''))
                        if exists(test_path):
                            coarse_mask_path = test_path
                            break
            
            if exists(coarse_mask_path):
                try:
                    coarse_mask_3d, _ = load_sitk_mask(coarse_mask_path)
                    # Extract 2D mask for key slice
                    if key_slice_idx_offset < coarse_mask_3d.shape[0]:
                        coarse_mask_2d = coarse_mask_3d[key_slice_idx_offset, :, :] > 0.5
                        print(f"    Using coarse mask from nnUNet for initialization")
                except Exception as e:
                    print(f"    Warning: Could not load coarse mask: {e}")
        
        # Use autocast for CUDA (mixed precision), regular inference for CPU
        if device.type == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(img_resized, video_height, video_width)
                
                # Initialize with coarse mask if available, otherwise use bounding box
                if coarse_mask_2d is not None:
                    # Use coarse mask as initial prompt (better than just bbox)
                    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=key_slice_idx_offset,
                        obj_id=1,
                        mask=coarse_mask_2d,
                    )
                    print(f"    Initialized with coarse mask on slice {key_slice_idx_offset}")
                elif propagate_with_box:
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=key_slice_idx_offset,
                                                        obj_id=1,
                                                        box=bbox,
                                                    )
                    print(f"    Initialized with bounding box on slice {key_slice_idx_offset}")
                else: # gt
                    pass

                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    # Follow exact pattern from medsam2_infer_CT_lesion_npz_recist.py line 383
                    # out_mask_logits[0] shape is [1, H, W], need [0] after numpy conversion
                    mask_2d = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    
                    # Debug: print shape on first few frames
                    if out_frame_idx <= key_slice_idx_offset + 2:
                        print(f"    Frame {out_frame_idx}: out_mask_logits[0] shape: {out_mask_logits[0].shape}, mask_2d shape: {mask_2d.shape}, dtype: {mask_2d.dtype}")
                        print(f"    Frame {out_frame_idx}: mask_2d non-zero: {np.sum(mask_2d > 0)}")
                    
                    # Ensure mask_2d is 2D [H, W] boolean
                    if mask_2d.ndim != 2:
                        print(f"    Warning: mask_2d has {mask_2d.ndim} dimensions, expected 2. Shape: {mask_2d.shape}")
                        continue
                    
                    mask_2d = mask_2d.astype(bool)
                    
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
                # Re-initialize for reverse propagation
                if coarse_mask_2d is not None:
                    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=key_slice_idx_offset,
                        obj_id=1,
                        mask=coarse_mask_2d,
                    )
                elif propagate_with_box:
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=key_slice_idx_offset,
                                                        obj_id=1,
                                                        box=bbox,
                                                    )
                else: # gt
                    pass

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
                    # Apply mask to segmentation (mask_2d should be [H, W] boolean)
                    segs_3D[out_frame_idx, mask_2d] = 1
                predictor.reset_state(inference_state)
        else:
            # CPU inference without autocast
            with torch.inference_mode():
                inference_state = predictor.init_state(img_resized, video_height, video_width)
                
                # Initialize with coarse mask if available, otherwise use bounding box
                if coarse_mask_2d is not None:
                    # Use coarse mask as initial prompt (better than just bbox)
                    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=key_slice_idx_offset,
                        obj_id=1,
                        mask=coarse_mask_2d,
                    )
                    print(f"    Initialized with coarse mask on slice {key_slice_idx_offset}")
                elif propagate_with_box:
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=key_slice_idx_offset,
                                                        obj_id=1,
                                                        box=bbox,
                                                    )
                    print(f"    Initialized with bounding box on slice {key_slice_idx_offset}")
                else: # gt
                    pass

                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    # Follow exact pattern from medsam2_infer_CT_lesion_npz_recist.py line 383
                    # out_mask_logits[0] shape is [1, H, W], need [0] after numpy conversion
                    mask_2d = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    
                    # Debug: print shape on first few frames
                    if out_frame_idx <= key_slice_idx_offset + 2:
                        print(f"    Frame {out_frame_idx}: out_mask_logits[0] shape: {out_mask_logits[0].shape}, mask_2d shape: {mask_2d.shape}, dtype: {mask_2d.dtype}")
                        print(f"    Frame {out_frame_idx}: mask_2d non-zero: {np.sum(mask_2d > 0)}")
                    
                    # Ensure mask_2d is 2D [H, W] boolean
                    if mask_2d.ndim != 2:
                        print(f"    Warning: mask_2d has {mask_2d.ndim} dimensions, expected 2. Shape: {mask_2d.shape}")
                        continue
                    
                    mask_2d = mask_2d.astype(bool)
                    
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
                # Re-initialize for reverse propagation
                if coarse_mask_2d is not None:
                    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=key_slice_idx_offset,
                        obj_id=1,
                        mask=coarse_mask_2d,
                    )
                elif propagate_with_box:
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=key_slice_idx_offset,
                                                        obj_id=1,
                                                        box=bbox,
                                                    )
                else: # gt
                    pass

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
        sitk_mask = sitk.GetImageFromArray(segs_3D.astype(np.uint8))
        sitk_mask.CopyInformation(nii_image)
        
        # Save segmentation
        if use_csv:
            save_seg_name = nii_fname.split('.nii.gz')[0] + f'_k{key_slice_idx}_mask.nii.gz'
            dicom_windows_str = f"{lower_bound}, {upper_bound}"
        else:
            save_seg_name = nii_fname.split('.nii.gz')[0] + f'_{label_name}_mask.nii.gz'
            dicom_windows_str = f"{lower_bound}, {upper_bound}"
        
        # Save original CT image once per case (not the preprocessed version)
        if row_id == 0:
            # Save the original CT image, not the preprocessed one
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



