# Workflow Verification Checklist

## ✅ Complete Workflow

### Step 1: nnUNet Segmentation → `nnunet_output/`
- Input: `ct_images/*.nii.gz`
- Output: `nnunet_output/*.nii.gz` (multi-label segmentation with labels 1-7)
- ✅ Verified: Script expects nnUNet outputs in this directory

### Step 2: Generate Prompts → `prompts_out/`
**Script:** `nnunet_to_medsam2_prompts.py`

- Input: 
  - `nnunet_output/*.nii.gz` (nnUNet segmentations)
  - `ct_images/*.nii.gz` (original CT images)
- Output: 
  - `prompts_out/{case_id}.json` (bounding boxes, centroids, paths)
  - `prompts_out/{case_id}_{label_name}_mask.nii.gz` (coarse dilated masks)
- ✅ Verified:
  - Extracts largest connected component ✓
  - Creates dilated coarse masks ✓
  - Extracts 3D bounding boxes ✓
  - Calculates centroids ✓
  - Saves JSON with all prompt info ✓

### Step 3: MedSAM2 Inference → `medsam2_results/`
**Script:** `medsam2_infer_3D_CT_modified.py`

- Input:
  - `ct_images/*.nii.gz` (from JSON `image_path`)
  - `prompts_out/{case_id}.json` (prompts with bboxes and coarse masks)
- Output:
  - `medsam2_results/{case_id}_{label_name}_mask.nii.gz` (refined individual masks)
  - `medsam2_results/{case_id}_img.nii.gz` (original CT image, saved once per case)
- ✅ Verified:
  - Reads JSON files from `prompts_dir` ✓
  - Uses `image_path` from JSON to load CT images ✓
  - Extracts bounding boxes from JSON ✓
  - Loads coarse masks for initialization ✓
  - Saves to `pred_save_dir` (medsam2_results) ✓
  - Saves original CT images (not preprocessed) ✓
  - Proper mask indexing fixed ✓

### Step 4: Stitch Masks → `medsam2_results/`
**Script:** `stitch_medsam2_segmentations.py`

- Input:
  - `medsam2_results/{case_id}_{label_name}_mask.nii.gz` (individual refined masks)
  - `ct_images/{case_id}.nii.gz` (reference for registration)
- Output:
  - `medsam2_results/{case_id}_seg.nii.gz` (combined multi-label segmentation)
- ✅ Verified:
  - Finds masks by label name ✓
  - Combines with correct label IDs (0=bg, 1=LV, 2=RV, etc.) ✓
  - Preserves spatial registration ✓
  - Output format matches nnUNet (same label convention) ✓

## File Paths Summary

```
ct_images/                    # Input CT images
  └── ct_1023.nii.gz
  └── ct_1028.nii.gz
  └── ct_1030.nii.gz

nnunet_output/                # nnUNet segmentations (Step 1 output)
  └── ct_1023.nii.gz          # Multi-label (0-7)
  └── ct_1028.nii.gz
  └── ct_1030.nii.gz

prompts_out/                  # Generated prompts (Step 2 output)
  └── ct_1023.json            # Prompt metadata
  └── ct_1023_LV_mask.nii.gz  # Coarse masks
  └── ct_1023_RV_mask.nii.gz
  └── ...

medsam2_results/              # MedSAM2 outputs (Step 3 & 4)
  └── ct_1023_LV_mask.nii.gz  # Refined individual masks
  └── ct_1023_RV_mask.nii.gz
  └── ...
  └── ct_1023_seg.nii.gz      # Final combined (Step 4 output)
```

## Potential Issues & Fixes

### Issue 1: Notebook uses MedSAM2 directory script
**Fix:** Updated notebook to use `medsam2_infer_3D_CT_modified.py` from root, or copy it to `MedSAM2/medsam2_infer_3D_CT.py`

### Issue 2: Config file path
**Fix:** Added `--cfg` argument to notebook cell pointing to MedSAM2 config

### Issue 3: Import paths in modified script
**Check:** Script imports from `sam2.build_sam` - assumes MedSAM2 is installed or in path

## Testing Checklist

- [ ] Step 1: nnUNet produces segmentations in `nnunet_output/`
- [ ] Step 2: Prompt generation creates JSON + masks in `prompts_out/`
- [ ] Step 3: MedSAM2 inference reads prompts and saves to `medsam2_results/`
- [ ] Step 4: Stitching combines masks into final `{case_id}_seg.nii.gz`
- [ ] Final output has correct label values (0-7)
- [ ] Final output has same registration as original CT

