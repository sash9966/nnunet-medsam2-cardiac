#!/usr/bin/env python3
"""
Stitch together individual MedSAM2 segmentation masks into a single multi-label segmentation file.

This script:
1. Reads individual mask files (e.g., ct_1023_LV_mask.nii.gz, ct_1023_RV_mask.nii.gz, etc.)
2. Combines them into a single segmentation with proper label IDs
3. Ensures the output has the same registration/affine as the original CT image
4. Saves the combined segmentation
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm

# Label mapping (same as in nnunet_to_medsam2_prompts.py)
# 0 = Background
# 1 = LV (Left Ventricle)
# 2 = RV (Right Ventricle)
# 3 = LA (Left Atrium)
# 4 = RA (Right Atrium)
# 5 = Myo (Myocardium)
# 6 = Aorta
# 7 = Pulmonary
LABEL_NAMES = {
    1: "LV",
    2: "RV",
    3: "LA",
    4: "RA",
    5: "Myo",
    6: "Aorta",
    7: "Pulmonary"
}

# Reverse mapping: label_name -> label_id
LABEL_NAME_TO_ID = {name: label_id for label_id, name in LABEL_NAMES.items()}


def load_nifti(nifti_path):
    """Load a NIfTI file and return data and NIfTI object."""
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    return data, nii


def load_sitk(sitk_path):
    """Load a NIfTI file using SimpleITK and return data and image object."""
    img = sitk.ReadImage(sitk_path)
    data = sitk.GetArrayFromImage(img)
    return data, img


def find_case_masks(masks_dir, case_id):
    """Find all mask files for a given case ID."""
    masks_dir = Path(masks_dir)
    mask_files = {}
    
    for label_id, label_name in LABEL_NAMES.items():
        mask_pattern = f"{case_id}_{label_name}_mask.nii.gz"
        mask_path = masks_dir / mask_pattern
        if mask_path.exists():
            mask_files[label_id] = mask_path
        else:
            # Try alternative naming (in case of different format)
            alt_pattern = f"{case_id}*{label_name}*mask*.nii.gz"
            alt_matches = list(masks_dir.glob(alt_pattern))
            if alt_matches:
                mask_files[label_id] = alt_matches[0]
    
    return mask_files


def get_reference_image(case_id, reference_dir=None, masks_dir=None):
    """
    Get reference image for registration.
    Priority:
    1. Original CT from reference_dir
    2. Preprocessed image from masks_dir (_img.nii.gz)
    3. First available mask (for affine only)
    """
    reference = None
    reference_path = None
    
    # Try original CT first
    if reference_dir:
        ref_path = Path(reference_dir) / f"{case_id}.nii.gz"
        if ref_path.exists():
            reference_path = ref_path
            reference, _ = load_sitk(str(ref_path))
            print(f"  Using original CT as reference: {reference_path}")
    
    # Try preprocessed image
    if reference is None and masks_dir:
        img_path = Path(masks_dir) / f"{case_id}_img.nii.gz"
        if img_path.exists():
            reference_path = img_path
            reference, _ = load_sitk(str(img_path))
            print(f"  Using preprocessed image as reference: {img_path}")
    
    # Fall back to first mask (for affine only)
    if reference is None and masks_dir:
        masks_dir = Path(masks_dir)
        mask_files = list(masks_dir.glob(f"{case_id}_*_mask.nii.gz"))
        if mask_files:
            reference_path = mask_files[0]
            reference, _ = load_sitk(str(reference_path))
            print(f"  Using first mask as reference (affine only): {reference_path}")
    
    return reference, reference_path


def combine_masks(mask_files, reference_shape=None, reference_affine=None):
    """
    Combine individual mask files into a single multi-label segmentation.
    
    Args:
        mask_files: Dict mapping label_id -> mask_file_path
        reference_shape: Target shape for output (from reference image)
        reference_affine: Target affine matrix (from reference image)
    
    Returns:
        combined_seg: Combined segmentation array with label IDs
        reference_shape: Shape of the combined segmentation
        reference_affine: Affine matrix from reference
    """
    combined_seg = None
    first_mask_shape = None
    first_affine = None
    
    # Load all masks and determine reference shape/affine
    mask_data = {}
    for label_id, mask_path in mask_files.items():
        if mask_path.exists():
            try:
                # Try SimpleITK first (preserves spatial info better)
                mask_array, mask_img = load_sitk(str(mask_path))
                mask_data[label_id] = {
                    'data': mask_array,
                    'img': mask_img,
                    'shape': mask_array.shape
                }
                
                # Store first mask's shape and affine as reference
                if first_mask_shape is None:
                    first_mask_shape = mask_array.shape
                    # Get affine from SimpleITK image
                    spacing = mask_img.GetSpacing()
                    origin = mask_img.GetOrigin()
                    direction = mask_img.GetDirection()
                    # Convert to numpy affine matrix (4x4)
                    direction_array = np.array(direction).reshape(3, 3)
                    affine = np.eye(4)
                    affine[:3, :3] = direction_array * np.array(spacing).reshape(3, 1)
                    affine[:3, 3] = origin
                    first_affine = affine
            except Exception as e:
                print(f"  Warning: Could not load {mask_path} with SimpleITK: {e}")
                try:
                    # Fall back to nibabel
                    mask_array, mask_nii = load_nifti(str(mask_path))
                    mask_data[label_id] = {
                        'data': mask_array,
                        'nii': mask_nii,
                        'shape': mask_array.shape
                    }
                    if first_mask_shape is None:
                        first_mask_shape = mask_array.shape
                        first_affine = mask_nii.affine
                except Exception as e2:
                    print(f"  Error: Could not load {mask_path}: {e2}")
                    continue
    
    if not mask_data:
        return None, None, None
    
    # Use reference shape if provided, otherwise use first mask shape
    target_shape = reference_shape if reference_shape is not None else first_mask_shape
    target_affine = reference_affine if reference_affine is not None else first_affine
    
    # Initialize combined segmentation
    combined_seg = np.zeros(target_shape, dtype=np.uint8)
    
    # Combine masks with proper label IDs
    # Process in label ID order (1=LV, 2=RV, 3=LA, etc.)
    # Background (0) is already set by initialization
    for label_id in sorted(mask_data.keys()):
        mask_info = mask_data[label_id]
        mask_array = mask_info['data']
        label_name = LABEL_NAMES.get(label_id, f"label_{label_id}")
        
        # Check if shapes match
        if mask_array.shape != target_shape:
            print(f"  Warning: Mask {label_id} ({label_name}) shape {mask_array.shape} "
                  f"doesn't match target shape {target_shape}. Resizing...")
            # Simple resizing (nearest neighbor)
            zoom_factors = [t/s for t, s in zip(target_shape, mask_array.shape)]
            mask_array = zoom(mask_array, zoom_factors, order=0)
        
        # Binarize mask: any non-zero value becomes this label
        # This ensures masks with values 0/1 or 0/255 both work correctly
        binary_mask = (mask_array > 0.5).astype(bool)
        
        # Assign label ID where mask is active
        # Later labels will overwrite earlier ones in case of overlap
        combined_seg[binary_mask] = label_id
        
        # Print info about this label
        num_voxels = np.sum(binary_mask)
        if num_voxels > 0:
            print(f"    Label {label_id} ({label_name}): {num_voxels} voxels")
    
    return combined_seg, target_shape, target_affine


def save_combined_segmentation(combined_seg, output_path, reference_img_path=None, reference_shape=None):
    """
    Save combined segmentation with proper registration.
    
    Args:
        combined_seg: Combined segmentation array
        output_path: Path to save output
        reference_img_path: Path to reference image (for copying spatial info)
        reference_shape: Shape of reference (if reference_img_path not available)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to use reference image for spatial information
    if reference_img_path and Path(reference_img_path).exists():
        try:
            # Use SimpleITK to preserve spatial information
            ref_img = sitk.ReadImage(str(reference_img_path))
            
            # Ensure shapes match
            ref_shape = sitk.GetArrayFromImage(ref_img).shape
            if combined_seg.shape != ref_shape:
                print(f"  Resizing segmentation from {combined_seg.shape} to {ref_shape}")
                zoom_factors = [r/s for r, s in zip(ref_shape, combined_seg.shape)]
                combined_seg = zoom(combined_seg, zoom_factors, order=0)
            
            # Create output image with same spatial info as reference
            output_img = sitk.GetImageFromArray(combined_seg.astype(np.uint8))
            output_img.CopyInformation(ref_img)
            sitk.WriteImage(output_img, str(output_path))
            print(f"  Saved with reference spatial info: {output_path}")
            return
        except Exception as e:
            print(f"  Warning: Could not use reference image, using mask info: {e}")
    
    # Fall back: use first available mask or create new image
    # Try to find a mask file to copy spatial info from
    masks_dir = output_path.parent
    case_id = output_path.stem.replace('_seg', '').replace('.nii', '')
    mask_files = list(masks_dir.glob(f"{case_id}_*_mask.nii.gz"))
    
    if mask_files:
        try:
            ref_mask = sitk.ReadImage(str(mask_files[0]))
            output_img = sitk.GetImageFromArray(combined_seg.astype(np.uint8))
            output_img.CopyInformation(ref_mask)
            sitk.WriteImage(output_img, str(output_path))
            print(f"  Saved with mask spatial info: {output_path}")
            return
        except Exception as e:
            print(f"  Warning: Could not use mask spatial info: {e}")
    
    # Last resort: save with nibabel
    try:
        # Create a basic affine (identity)
        affine = np.eye(4)
        nii = nib.Nifti1Image(combined_seg.astype(np.uint8), affine)
        nib.save(nii, str(output_path))
        print(f"  Saved with default affine: {output_path}")
    except Exception as e:
        print(f"  Error saving segmentation: {e}")


def process_case(case_id, masks_dir, output_dir, reference_dir=None):
    """Process a single case: combine masks and save."""
    print(f"\nProcessing case: {case_id}")
    
    # Find all masks for this case
    mask_files = find_case_masks(masks_dir, case_id)
    
    if not mask_files:
        print(f"  No mask files found for {case_id}")
        return False
    
    print(f"  Found {len(mask_files)} mask files:")
    for label_id, mask_path in sorted(mask_files.items()):
        print(f"    {label_id}: {LABEL_NAMES[label_id]} - {mask_path.name}")
    
    # Get reference image for registration
    reference, reference_path = get_reference_image(case_id, reference_dir, masks_dir)
    
    # Combine masks
    if reference is not None:
        reference_shape = reference.shape
        # Get affine from reference if possible
        try:
            ref_img = sitk.ReadImage(str(reference_path))
            # For SimpleITK, we'll copy info later
            reference_affine = None
        except:
            reference_affine = None
    else:
        reference_shape = None
        reference_affine = None
    
    combined_seg, final_shape, final_affine = combine_masks(
        mask_files, 
        reference_shape=reference_shape,
        reference_affine=reference_affine
    )
    
    if combined_seg is None:
        print(f"  Failed to combine masks for {case_id}")
        return False
    
    # Save combined segmentation
    output_path = Path(output_dir) / f"{case_id}_seg.nii.gz"
    save_combined_segmentation(
        combined_seg, 
        output_path,
        reference_img_path=reference_path,
        reference_shape=reference_shape
    )
    
    # Verify label values
    unique_labels = sorted(np.unique(combined_seg))
    print(f"  ✓ Combined segmentation saved: {output_path}")
    print(f"    Shape: {combined_seg.shape}")
    print(f"    Labels present: {unique_labels}")
    print(f"    Label mapping:")
    for label_val in unique_labels:
        if label_val == 0:
            print(f"      {label_val} = Background")
        else:
            label_name = LABEL_NAMES.get(label_val, f"Unknown({label_val})")
            num_voxels = np.sum(combined_seg == label_val)
            print(f"      {label_val} = {label_name} ({num_voxels} voxels)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stitch together individual MedSAM2 segmentation masks into a single multi-label segmentation"
    )
    parser.add_argument(
        '--masks_dir',
        type=str,
        required=True,
        help='Directory containing individual mask files (e.g., prompts_out/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save combined segmentations'
    )
    parser.add_argument(
        '--reference_dir',
        type=str,
        default=None,
        help='Directory with original CT images (for registration). If not provided, uses preprocessed images from masks_dir'
    )
    parser.add_argument(
        '--case_ids',
        type=str,
        nargs='+',
        default=None,
        help='Specific case IDs to process (e.g., ct_1023 ct_1028). If not provided, processes all cases found in masks_dir'
    )
    
    args = parser.parse_args()
    
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all case IDs
    if args.case_ids:
        case_ids = args.case_ids
    else:
        # Find all unique case IDs from mask files
        # Also check JSON files for case IDs
        json_files = list(masks_dir.glob("*.json"))
        case_ids = set()
        
        # First, try to get case IDs from JSON files
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'case_id' in data:
                        case_ids.add(data['case_id'])
            except:
                pass
        
        # Also extract from mask filenames as fallback
        if not case_ids:
            mask_files = list(masks_dir.glob("*_*_mask.nii.gz"))
            for mask_file in mask_files:
                # Extract case_id from filename like "ct_1023_LV_mask.nii.gz"
                # Pattern: {case_id}_{label_name}_mask.nii.gz
                # Remove .nii.gz extension and _mask suffix
                name_base = mask_file.name.replace('.nii.gz', '').replace('_mask', '')
                # Find the label name in the name_base
                for label_name in LABEL_NAMES.values():
                    if name_base.endswith(f'_{label_name}'):
                        case_id = name_base[:-len(f'_{label_name}')]
                        case_ids.add(case_id)
                        break
        
        case_ids = sorted(case_ids)
    
    print(f"Found {len(case_ids)} cases to process: {case_ids}")
    
    # Process each case
    success_count = 0
    for case_id in tqdm(case_ids, desc="Processing cases"):
        if process_case(case_id, masks_dir, output_dir, args.reference_dir):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successfully processed: {success_count}/{len(case_ids)} cases")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

