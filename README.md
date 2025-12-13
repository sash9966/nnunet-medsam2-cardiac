# nnUNet + MedSAM2: Cardiac CT Segmentation Pipeline

A complete pipeline for cardiac CT segmentation that combines nnUNet for initial segmentation with MedSAM2 for refinement using prompt-based segmentation.

## Overview

This project implements a two-stage segmentation approach:
1. **nnUNet**: Provides initial segmentations and generates prompts (bounding boxes, coarse masks)
2. **MedSAM2**: Refines the segmentations using the nnUNet prompts for improved accuracy

## Project Structure

```
.
├── ct_images/              # Input CT images
├── nnunet_output/          # nnUNet segmentation outputs
├── prompts_out/            # Generated prompts (JSON + coarse masks)
├── medsam2_results/        # Final MedSAM2 refined segmentations
├── nnunet_to_medsam2_prompts.py  # Generate prompts from nnUNet outputs
├── stitch_medsam2_segmentations.py  # Combine individual masks into multi-label segmentation
├── medsam2.ipynb           # Main workflow notebook
└── MedSAM2/                # MedSAM2 inference code (modified)
```

## Label Mapping

- **0**: Background
- **1**: LV (Left Ventricle)
- **2**: RV (Right Ventricle)
- **3**: LA (Left Atrium)
- **4**: RA (Right Atrium)
- **5**: Myo (Myocardium)
- **6**: Aorta
- **7**: Pulmonary

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- SimpleITK
- nibabel
- scipy
- tqdm
- nnUNetv2

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd nnUNetandMedSAM
```

2. Install MedSAM2 dependencies:
```bash
cd MedSAM2
pip install -e .
cd ..
```

3. Download MedSAM2 checkpoint:
   - Place `MedSAM2_CTLesion.pt` in `MedSAM2/checkpoints/`
   - See `MedSAM2/checkpoints/README.md` for download instructions

## Usage

### Step 1: Run nnUNet Segmentation

First, run nnUNet to get initial segmentations:

```bash
nnUNetv2_predict \
  -i /path/to/ct_images \
  -o nnunet_output \
  -d DatasetXXX \
  -c 3d_fullres \
  -f all
```

### Step 2: Generate Prompts from nnUNet Outputs

Extract bounding boxes and create coarse masks from nnUNet segmentations:

```bash
python3 nnunet_to_medsam2_prompts.py \
    --nnunet_output nnunet_output \
    --ct_dir ct_images \
    --out_dir prompts_out \
    --dilation_iters 3 \
    --bbox_padding 5
```

This creates:
- JSON files with bounding boxes and centroids for each case
- Coarse masks (dilated nnUNet segmentations) for initialization

### Step 3: Run MedSAM2 Inference

Refine segmentations using MedSAM2 with the generated prompts:

```bash
python3 MedSAM2/medsam2_infer_3D_CT.py \
    -i ct_images \
    -o medsam2_results \
    --prompts_dir prompts_out \
    --checkpoint MedSAM2/checkpoints/MedSAM2_CTLesion.pt
```

### Step 4: Stitch Individual Masks Together

Combine individual label masks into a single multi-label segmentation:

```bash
python3 stitch_medsam2_segmentations.py \
    --masks_dir medsam2_results \
    --output_dir medsam2_results \
    --reference_dir ct_images
```

Output: `{case_id}_seg.nii.gz` files with all labels combined.

## Workflow Notebook

See `medsam2.ipynb` for a complete interactive workflow with all steps.

## Key Features

- **Automatic device detection**: Uses GPU (CUDA) if available, falls back to CPU
- **Prompt-based refinement**: Uses nnUNet outputs as initialization for MedSAM2
- **3D propagation**: Propagates segmentations through entire 3D volumes
- **Multi-label support**: Handles all 7 cardiac structures simultaneously
- **Registration preservation**: Maintains spatial registration with original CT images

## Modified Files

### `MedSAM2/medsam2_infer_3D_CT.py`

Key modifications:
- Added JSON prompt support (reads bounding boxes from `nnunet_to_medsam2_prompts.py` output)
- Automatic device detection (GPU/CPU)
- Uses coarse masks from nnUNet for better initialization
- Saves original CT images (not preprocessed versions)
- Improved error handling and debug output

## Output Files

### Individual Masks (per label)
- `{case_id}_{label_name}_mask.nii.gz` - Individual refined masks for each label

### Combined Segmentation
- `{case_id}_seg.nii.gz` - Multi-label segmentation with all structures (0-7)

## Citation

If you use this code, please cite:
- nnUNet: https://github.com/MIC-DKFZ/nnUNet
- MedSAM2: https://github.com/bowang-lab/MedSAM2

## License

See individual component licenses:
- MedSAM2: See `MedSAM2/LICENSE`
- This pipeline: MIT License (modifications)

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Troubleshooting

### Empty Masks
- Check that prompts were generated correctly
- Verify bounding boxes are within image bounds
- Check device availability (GPU/CPU)

### Corrupted Images
- Ensure original CT images are being saved (not preprocessed versions)
- Check data types (should be original CT intensity values)

### Shape Mismatches
- Verify all input images have consistent dimensions
- Check that reference images match segmentation shapes

