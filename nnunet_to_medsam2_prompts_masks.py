import numpy as np
import SimpleITK as sitk
import json
from pathlib import Path
from scipy.ndimage import binary_dilation

def create_prompts_with_negatives(nnunet_seg_path, ct_path, output_dir, dilation_iters=3):
    """
    Create mask prompts with negative examples for each label.
    
    For each target label:
    - Positive mask: The dilated nnUNet mask for that label
    - Negative masks: All OTHER labels (to exclude them)
    """
    
    # Load nnUNet segmentation
    seg_sitk = sitk.ReadImage(str(nnunet_seg_path))
    seg_array = sitk.GetArrayFromImage(seg_sitk)  # Shape: (D, H, W)
    
    # Load CT for reference
    ct_sitk = sitk.ReadImage(str(ct_path))
    ct_array = sitk.GetArrayFromImage(ct_sitk)
    
    label_names = {
        1: "LV", 2: "RV", 3: "LA", 4: "RA",
        5: "Myo", 6: "Aorta", 7: "Pulmonary"
    }
    
    # Extract case_id - handle both ct_XXXX.nii.gz and ct_XXXX_seg.nii.gz
    case_id = Path(nnunet_seg_path).stem.replace('.nii', '').replace('_seg', '')
    output_case_dir = Path(output_dir) / case_id
    output_case_dir.mkdir(parents=True, exist_ok=True)
    
    prompts_data = {
        "case_id": case_id,
        "labels": {}
    }
    
    for label_id, label_name in label_names.items():
        if not np.any(seg_array == label_id):
            print(f"  Skipping {label_name} (label {label_id}) - not present")
            continue
            
        # Create positive mask (dilated for initialization)
        positive_mask = (seg_array == label_id).astype(np.uint8)
        
        # Dilate to expand slightly beyond nnUNet boundaries
        if dilation_iters > 0:
            struct = np.ones((3, 3, 3), dtype=bool)
            positive_mask = binary_dilation(positive_mask, structure=struct, 
                                           iterations=dilation_iters).astype(np.uint8)
        
        # Create negative masks (all OTHER labels combined)
        # This tells MedSAM: "segment the positive region, but DON'T include these areas"
        negative_mask = np.zeros_like(seg_array, dtype=np.uint8)
        for other_label in label_names.keys():
            if other_label != label_id:
                negative_mask = np.logical_or(negative_mask, seg_array == other_label)
        negative_mask = negative_mask.astype(np.uint8)
        
        # CRITICAL for VSDs: dilate negative masks to create buffer zones
        # This prevents MedSAM from "leaking" through defects
        if dilation_iters > 0:
            negative_mask = binary_dilation(negative_mask, structure=struct,
                                           iterations=dilation_iters).astype(np.uint8)
        
        # Save positive mask
        pos_mask_sitk = sitk.GetImageFromArray(positive_mask)
        pos_mask_sitk.CopyInformation(seg_sitk)
        pos_mask_path = output_case_dir / f"{label_name}_positive_mask.nii.gz"
        sitk.WriteImage(pos_mask_sitk, str(pos_mask_path))
        
        # Save negative mask
        neg_mask_sitk = sitk.GetImageFromArray(negative_mask)
        neg_mask_sitk.CopyInformation(seg_sitk)
        neg_mask_path = output_case_dir / f"{label_name}_negative_mask.nii.gz"
        sitk.WriteImage(neg_mask_sitk, str(neg_mask_path))
        
        # Calculate bounding box (still useful for initial region of interest)
        coords = np.argwhere(positive_mask > 0)
        if len(coords) == 0:
            continue
            
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Store prompt information
        prompts_data["labels"][label_name] = {
            "label_id": int(label_id),
            "positive_mask_path": str(pos_mask_path.name),
            "negative_mask_path": str(neg_mask_path.name),
            "bbox_3d": {
                "z": [int(z_min), int(z_max)],
                "y": [int(y_min), int(y_max)],
                "x": [int(x_min), int(x_max)]
            },
            "num_positive_voxels": int(positive_mask.sum()),
            "num_negative_voxels": int(negative_mask.sum())
        }
        
        print(f"  {label_name}: pos={positive_mask.sum()} voxels, "
              f"neg={negative_mask.sum()} voxels")
    
    # Save JSON
    json_path = output_case_dir / "prompts.json"
    with open(json_path, 'w') as f:
        json.dump(prompts_data, f, indent=2)
    
    return prompts_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnunet_output', required=True)
    parser.add_argument('--ct_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--dilation_iters', type=int, default=3)
    args = parser.parse_args()
    
    nnunet_dir = Path(args.nnunet_output)
    ct_dir = Path(args.ct_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try both naming patterns: *_seg.nii.gz and *.nii.gz
    seg_files = list(nnunet_dir.glob("*_seg.nii.gz"))
    if not seg_files:
        seg_files = list(nnunet_dir.glob("*.nii.gz"))
    
    for seg_file in sorted(seg_files):
        # Extract case_id - handle both ct_XXXX.nii.gz and ct_XXXX_seg.nii.gz
        # .stem removes .gz, so we need to also remove .nii
        case_id = seg_file.stem.replace('.nii', '').replace('_seg', '')
        ct_file = ct_dir / f"{case_id}_0000.nii.gz"  # Adjust pattern as needed
        
        if not ct_file.exists():
            print(f"Skipping {case_id}: CT not found")
            continue
            
        print(f"Processing {case_id}...")
        create_prompts_with_negatives(seg_file, ct_file, output_dir, 
                                     args.dilation_iters)