# Uncertainty-Guided Refinement Pipeline for imageCHD (CT) Dataset

This pipeline implements an uncertainty-guided refinement approach that leverages MedSAM2 only where nnU-Net is uncertain, while preserving nnU-Net's superior anatomical topology for CHD-specific defects like VSDs.

## Overview

The pipeline consists of three main scripts:

1. **`nnunet_uncertainty_analysis.py`** - Analyzes nnU-Net probability maps to identify high-confidence seeds and uncertainty zones
2. **`medsam2_infer_3D_CT_uncertainty_guided.py`** - Performs MedSAM2 inference using point prompts, with selective refinement based on structure type
3. **`stitch_medsam2_segmentations.py`** - Combines individual masks with optional uncertainty-guided weighted ensemble

## Pipeline Workflow

### Step 1: Uncertainty Analysis

Analyze nnU-Net probability maps to identify:
- **High-Confidence Seeds**: Voxels with probability > 0.95 for each label
- **Uncertainty Zones**: Regions with entropy above the threshold (default: 75th percentile)

```bash
python3 nnunet_uncertainty_analysis.py \
    --prob_dir nnunet_probabilities_Dataset1Ensemblefrom100DAepoch5foldvalidationtest \
    --prompts_dir prompts_out \
    --nnunet_seg_dir nnunet_probabilities_Dataset1Ensemblefrom100DAepoch5foldvalidationtest \
    --num_points 8 \
    --confidence_threshold 0.95 \
    --entropy_percentile 75.0
```

**Parameters:**
- `--prob_dir`: Directory containing `.npz` probability files from nnU-Net
- `--prompts_dir`: Directory containing JSON prompt files (will be updated)
- `--nnunet_seg_dir`: Directory with nnU-Net segmentation files (for VSD detection)
- `--num_points`: Number of points to sample from high-confidence seeds (default: 8)
- `--confidence_threshold`: Probability threshold for high-confidence seeds (default: 0.95)
- `--entropy_percentile`: Percentile to use as entropy threshold (default: 75.0)

**Output:**
- Updates JSON files in `prompts_dir` with `uncertainty_analysis` field containing:
  - `high_confidence_seeds`: List of [z, y, x] coordinates
  - `uncertainty_zones`: List of uncertainty zone coordinates
  - `is_high_contrast`: Boolean flag for high-contrast structures (LV, RV, Aorta, Pulmonary)
  - `is_topologically_complex`: Boolean flag for complex structures (Myocardium)
  - `has_vsd`: Boolean flag indicating VSD detection

### Step 2: Uncertainty-Guided MedSAM2 Inference

Perform MedSAM2 inference with point prompts, selectively refining structures:

- **High-Contrast Structures** (LV, RV, Aorta, Pulmonary): Use MedSAM2 refinement with point prompts
- **Topologically Complex Structures** (Myocardium, VSD areas): Skip MedSAM2 refinement, use nnU-Net segmentation directly

```bash
python3 medsam2_infer_3D_CT_uncertainty_guided.py \
    --checkpoint MedSAM2/checkpoints/MedSAM2_CTLesion.pt \
    --cfg sam2/configs/sam2.1_hiera_t512.yaml \
    -i ct_images \
    --prompts_dir prompts_out \
    --nnunet_seg_dir nnunet_outputfull3DDA51000epochs \
    -o medsam2_results_uncertainty
```

**Parameters:**
- `--checkpoint`: Path to MedSAM2 checkpoint
- `--cfg`: Path to MedSAM2 config file
- `-i, --imgs_path`: Directory containing CT images
- `--prompts_dir`: Directory containing JSON prompt files with uncertainty analysis
- `--nnunet_seg_dir`: Directory with nnU-Net segmentation files (for fallback)
- `-o, --pred_save_dir`: Output directory for MedSAM2 masks

**Output:**
- Individual mask files: `{case_id}_{label_name}_mask.nii.gz`
- For topologically complex structures, uses nnU-Net segmentation directly
- For high-contrast structures, uses MedSAM2 with point prompts from high-confidence seeds

### Step 3: Stitch Segmentations with Uncertainty-Guided Ensemble

Combine individual masks into multi-label segmentation with optional weighted ensemble:

**Standard Combination:**
```bash
python3 stitch_medsam2_segmentations.py \
    --masks_dir medsam2_results_uncertainty \
    --output_dir final_segmentations \
    --reference_dir ct_images
```

**Uncertainty-Guided Weighted Ensemble:**
```bash
python3 stitch_medsam2_segmentations.py \
    --masks_dir medsam2_results_uncertainty \
    --output_dir final_segmentations \
    --reference_dir ct_images \
    --nnunet_seg_dir nnunet_outputfull3DDA51000epochs \
    --prompts_dir prompts_out \
    --use_uncertainty_ensemble
```

**Parameters:**
- `--masks_dir`: Directory containing individual MedSAM2 mask files
- `--output_dir`: Output directory for combined segmentations
- `--reference_dir`: Directory with original CT images (for registration)
- `--nnunet_seg_dir`: Directory with nnU-Net segmentation files (required for ensemble)
- `--prompts_dir`: Directory with JSON prompt files containing uncertainty zones (required for ensemble)
- `--use_uncertainty_ensemble`: Enable uncertainty-guided weighted ensemble

**Ensemble Logic:**
When `--use_uncertainty_ensemble` is enabled:
- **Final Mask = (nnU-Net Core) + (MedSAM2 output restricted to Uncertainty Zone)**
- Preserves nnU-Net's high-confidence regions
- Applies MedSAM2 refinement only in uncertainty zones
- Prevents edge-leakage in topologically complex structures

## Structure Classification

### High-Contrast Structures
- **LV** (Left Ventricle)
- **RV** (Right Ventricle)
- **Aorta**
- **Pulmonary**

These structures benefit from MedSAM2 refinement using point prompts from high-confidence seeds.

### Topologically Complex Structures
- **Myo** (Myocardium)
- **VSD areas** (Ventricular Septal Defects)

These structures skip MedSAM2 refinement and retain nnU-Net segmentation to preserve anatomical topology.

## File Naming Conventions

- CT images: `{case_id}_0000.nii.gz` (e.g., `ct_1004_0000.nii.gz`)
- nnU-Net segmentations: `{case_id}.nii.gz` (e.g., `ct_1004.nii.gz`)
- nnU-Net probabilities: `{case_id}.npz` (e.g., `ct_1004.npz`)
- MedSAM2 masks: `{case_id}_{label_name}_mask.nii.gz` (e.g., `ct_1004_LV_mask.nii.gz`)
- Combined segmentations: `{case_id}_medsamrefined_uncertainty_seg.nii.gz`

## Example: Complete Pipeline

```bash
# Step 1: Analyze uncertainty
python3 nnunet_uncertainty_analysis.py \
    --prob_dir nnunet_probabilities_Dataset1Ensemblefrom100DAepoch5foldvalidationtest \
    --prompts_dir prompts_out \
    --nnunet_seg_dir nnunet_probabilities_Dataset1Ensemblefrom100DAepoch5foldvalidationtest

# Step 2: Run MedSAM2 inference
python3 medsam2_infer_3D_CT_uncertainty_guided.py \
    --checkpoint MedSAM2/checkpoints/MedSAM2_CTLesion.pt \
    --cfg sam2/configs/sam2.1_hiera_t512.yaml \
    -i ct_images \
    --prompts_dir prompts_out \
    --nnunet_seg_dir nnunet_outputfull3DDA51000epochs \
    -o medsam2_results_uncertainty

# Step 3: Stitch with ensemble
python3 stitch_medsam2_segmentations.py \
    --masks_dir medsam2_results_uncertainty \
    --output_dir final_segmentations \
    --reference_dir ct_images \
    --nnunet_seg_dir nnunet_outputfull3DDA51000epochs \
    --prompts_dir prompts_out \
    --use_uncertainty_ensemble
```

## Key Features

1. **Point Prompts**: Uses 5-10 positive points from high-confidence seeds instead of bounding boxes
2. **Selective Refinement**: Only refines high-contrast structures, preserves nnU-Net for complex structures
3. **Uncertainty Zones**: MedSAM2 refinement restricted to regions where nnU-Net is uncertain
4. **Weighted Ensemble**: Combines nnU-Net core with MedSAM2 refinement in uncertainty zones
5. **VSD Protection**: Skips MedSAM2 refinement for VSD areas to prevent edge-leakage

## Notes

- The pipeline ensures the `_0000` naming convention is strictly followed for image-mask mapping
- Uncertainty zones are limited to 1000 coordinates in JSON files to keep file sizes manageable
- For topologically complex structures, the pipeline falls back to nnU-Net segmentation if MedSAM2 masks are not available
- The entropy threshold is calculated per-case using the specified percentile
