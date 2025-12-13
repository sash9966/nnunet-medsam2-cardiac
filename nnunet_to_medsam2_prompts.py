import os
import json
from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, label
from tqdm import tqdm

LABEL_NAMES = {
    1: "LV",
    2: "RV",
    3: "LA",
    4: "RA",
    5: "Myo",
    6: "Aorta",
    7: "Pulmonary"
}

def load_seg(seg_path):
    img = nib.load(seg_path)
    return img.get_fdata().astype(np.uint8), img

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_largest_component(mask):
    """
    Extract the largest connected component from a binary mask.
    Returns the mask containing only the largest component.
    """
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return mask
    
    # Find the largest component
    component_sizes = []
    for i in range(1, num_features + 1):
        component_sizes.append((i, np.sum(labeled_mask == i)))
    
    if not component_sizes:
        return mask
    
    largest_component_id = max(component_sizes, key=lambda x: x[1])[0]
    largest_component_mask = labeled_mask == largest_component_id
    
    return largest_component_mask

def extract_bbox(mask, padding=5, shape=None):
    """
    Extract bounding box from mask with optional padding.
    
    Args:
        mask: Binary mask
        padding: Voxels to pad in each direction (default: 5)
        shape: Shape of the volume to clip bbox to (default: None, uses mask.shape)
    
    Returns:
        [zmin, ymin, xmin, zmax, ymax, xmax] with padding applied
    """
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    
    # Apply padding
    if shape is None:
        shape = mask.shape
    
    zmin = max(0, int(zmin) - padding)
    ymin = max(0, int(ymin) - padding)
    xmin = max(0, int(xmin) - padding)
    zmax = min(shape[0] - 1, int(zmax) + padding)
    ymax = min(shape[1] - 1, int(ymax) + padding)
    xmax = min(shape[2] - 1, int(xmax) + padding)
    
    return [zmin, ymin, xmin, zmax, ymax, xmax]

def main(args):

    nnunet_out = Path(args.nnunet_output)
    ct_dir = Path(args.ct_dir)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)
    prompts_jsonl = open(out_dir / "prompts_index.jsonl", "w")

    seg_files = sorted([f for f in nnunet_out.glob("*.nii.gz")])

    for seg_file in tqdm(seg_files, desc="Processing nnUNet outputs"):
        seg_arr, seg_nifti = load_seg(seg_file)
        case_id = seg_file.stem.replace("_seg", "").replace(".nii", "")

        # find original CT
        ct_candidates = list(ct_dir.glob(f"{case_id}*.nii.gz"))
        if len(ct_candidates) == 0:
            print(f"[WARN] No CT found for {case_id}")
            continue
        ct_path = ct_candidates[0]

        case_prompts = {
            "case_id": case_id,
            "image_path": str(ct_path),
            "nnunet_segmentation": str(seg_file),
            "prompts": []
        }

        for label_id, name in LABEL_NAMES.items():
            mask = seg_arr == label_id
            if np.sum(mask) == 0:
                continue

            # Extract largest connected component to avoid issues with disjoint regions
            largest_component = get_largest_component(mask)
            
            # Skip if largest component is empty (shouldn't happen, but safety check)
            if np.sum(largest_component) == 0:
                print(f"[WARN] Empty largest component for {case_id} label {name}")
                continue

            # create dilated coarse mask from largest component
            coarse = binary_dilation(largest_component, iterations=args.dilation_iters)

            coarse_path = out_dir / f"{case_id}_{name}_mask.nii.gz"
            coarse_nifti = nib.Nifti1Image(
                coarse.astype(np.uint8),
                seg_nifti.affine,
                seg_nifti.header
            )
            nib.save(coarse_nifti, coarse_path)

            # Extract bbox from largest component with safety padding
            bbox = extract_bbox(largest_component, padding=args.bbox_padding, shape=seg_arr.shape)
            if bbox is None:
                print(f"[WARN] Could not extract bbox for {case_id} label {name}")
                continue
            
            # Centroid from largest component only
            centroid = np.argwhere(largest_component).mean(axis=0).tolist()

            case_prompts["prompts"].append({
                "label_id": label_id,
                "label_name": name,
                "coarse_mask_path": str(coarse_path),
                "voxel_bbox": bbox,
                "centroid_voxel": [float(x) for x in centroid]
            })

        # save per-case JSON
        json_path = out_dir / f"{case_id}.json"
        with open(json_path, "w") as f:
            json.dump(case_prompts, f, indent=2)

        # also write to JSONL index
        prompts_jsonl.write(json.dumps(case_prompts) + "\n")

    prompts_jsonl.close()
    print(f"\n✔ Prompt generation complete.")
    print(f"  → JSON + coarse masks written to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet_output", required=True,
        help="Folder containing nnU-Net predicted segmentations (*.nii.gz).")
    parser.add_argument("--ct_dir", required=True,
        help="Folder with original CT NIfTI files.")
    parser.add_argument("--out_dir", required=True,
        help="Folder where prompts + masks are written.")
    parser.add_argument("--dilation_iters", type=int, default=3,
        help="How much to dilate nnU-Net masks to create coarse prompts.")
    parser.add_argument("--bbox_padding", type=int, default=5,
        help="Safety padding (in voxels) to add around bounding boxes. Default: 5")
    args = parser.parse_args()
    main(args)
