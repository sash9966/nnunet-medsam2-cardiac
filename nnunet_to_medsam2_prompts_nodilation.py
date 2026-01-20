import numpy as np
import SimpleITK as sitk
import json
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion

def create_prompts_with_negatives(nnunet_seg_path, ct_path, output_dir, positive_dilation=0, negative_erosion=2):
    """
    Create mask prompts with negative examples for each label.
    
    NO-DILATION APPROACH:
    - Positive mask: Minimal or NO dilation (prevents fattening)
    - Negative mask: ERODED instead of dilated (creates safety buffer without expansion)
    
    For each target label:
    - Positive mask: The nnUNet mask for that label (optionally minimally dilated)
    - Negative masks: All OTHER labels (eroded to create buffer)
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
        "image_path": str(ct_path),  # Store CT image path for inference
        "labels": {}
    }
    
    struct = np.ones((3, 3, 3), dtype=bool)
    
    for label_id, label_name in label_names.items():
        if not np.any(seg_array == label_id):
            print(f"  Skipping {label_name} (label {label_id}) - not present")
            continue
            
        # Create positive mask - NO DILATION or minimal dilation
        positive_mask = (seg_array == label_id).astype(np.uint8)
        
        # FIX 1: Minimal or NO dilation on positive masks
        if positive_dilation > 0:
            positive_mask = binary_dilation(positive_mask, structure=struct, 
                                           iterations=positive_dilation).astype(np.uint8)
        # If positive_dilation == 0, keep mask as-is (no dilation)
        
        # Create negative masks (all OTHER labels combined)
        # This tells MedSAM: "segment the positive region, but DON'T include these areas"
        negative_mask = np.zeros_like(seg_array, dtype=np.uint8)
        for other_label in label_names.keys():
            if other_label != label_id:
                negative_mask = np.logical_or(negative_mask, seg_array == other_label)
        negative_mask = negative_mask.astype(np.uint8)
        
        # FIX 2: ERODE negative masks instead of dilate
        # This creates a safety buffer without expanding the negative region
        # Prevents MedSAM from segmenting into other structures
        if negative_erosion > 0:
            negative_mask = binary_erosion(negative_mask, structure=struct,
                                          iterations=negative_erosion).astype(np.uint8)
        
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
        
        print(f"  {label_name}: pos={positive_mask.sum()} voxels (dilation={positive_dilation}), "
              f"neg={negative_mask.sum()} voxels (erosion={negative_erosion})")
    
    # Save JSON
    json_path = output_case_dir / "prompts.json"
    with open(json_path, 'w') as f:
        json.dump(prompts_data, f, indent=2)
    
    return prompts_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate MedSAM2 prompts with NO-DILATION approach (fixes fattening)"
    )
    parser.add_argument('--nnunet_output', required=True,
                        help='Directory containing nnUNet segmentation outputs')
    parser.add_argument('--ct_dir', required=True,
                        help='Directory containing CT images')
    parser.add_argument('--out_dir', default='prompts_nodilation',
                        help='Output directory for prompts (default: prompts_nodilation)')
    parser.add_argument('--positive_dilation', type=int, default=0,
                        help='Dilation iterations for positive mask (0=none, 1=minimal, default=0)')
    parser.add_argument('--negative_erosion', type=int, default=2,
                        help='Erosion iterations for negative mask (default=2)')
    parser.add_argument('--case_id', type=str, default=None,
                        help='Process only this specific case ID (e.g., ct_1023). If not provided, processes all cases.')
    args = parser.parse_args()
    
    nnunet_dir = Path(args.nnunet_output)
    ct_dir = Path(args.ct_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try both naming patterns: *_seg.nii.gz and *.nii.gz
    seg_files = list(nnunet_dir.glob("*_seg.nii.gz"))
    if not seg_files:
        seg_files = list(nnunet_dir.glob("*.nii.gz"))
    
    # Filter to single case if specified
    if args.case_id:
        case_id_filter = args.case_id.replace('.nii.gz', '').replace('.nii', '').replace('_seg', '')
        seg_files = [f for f in seg_files if case_id_filter in f.stem]
        if not seg_files:
            print(f"ERROR: No segmentation file found for case {args.case_id}")
            import sys
            sys.exit(1)
        print(f"TEST MODE: Processing only case {args.case_id}")
    else:
        print(f"Found {len(seg_files)} segmentation files")
    
    print(f"Positive mask dilation: {args.positive_dilation} iterations")
    print(f"Negative mask erosion: {args.negative_erosion} iterations")
    print(f"Output directory: {output_dir}")
    
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
                                     args.positive_dilation, args.negative_erosion)

