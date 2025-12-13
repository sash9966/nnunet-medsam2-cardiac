# Setup Instructions for GitHub Repository

## Initial Git Setup

1. **Initialize and commit your code:**
```bash
git init
git add .gitignore README.md
git add nnunet_to_medsam2_prompts.py stitch_medsam2_segmentations.py medsam2.ipynb
git add MedSAM2/medsam2_infer_3D_CT.py
git commit -m "Initial commit: nnUNet + MedSAM2 cardiac segmentation pipeline"
```

2. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Choose a repository name (e.g., `cardiac-segmentation-pipeline`)
   - Don't initialize with README (we already have one)
   - Click "Create repository"

3. **Link your local repository to GitHub:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## What's Included

The repository includes:
- ✅ Core pipeline scripts (`nnunet_to_medsam2_prompts.py`, `stitch_medsam2_segmentations.py`)
- ✅ Modified MedSAM2 inference script
- ✅ Workflow notebook
- ✅ Documentation (README.md)
- ✅ .gitignore (excludes large data files and checkpoints)

## What's Excluded (via .gitignore)

- Large medical images (`.nii.gz`, `.nii`, `.npz`)
- Model checkpoints (`.pt`, `.pth`)
- Output directories (`medsam2_results/`, `prompts_out/`, `nnunet_output/`)
- Python cache files (`__pycache__/`, `.pyc`)
- Jupyter checkpoints

## Adding MedSAM2 Submodule (Optional)

If you want to track the full MedSAM2 repository as a submodule:

```bash
# Remove the current MedSAM2 directory (backup first!)
git rm -r --cached MedSAM2
git submodule add https://github.com/bowang-lab/MedSAM2.git MedSAM2
git commit -m "Add MedSAM2 as submodule"
```

Then apply your modifications to `MedSAM2/medsam2_infer_3D_CT.py` and commit them.

## Notes

- Large data files should be stored separately (e.g., using Git LFS or external storage)
- Model checkpoints should be downloaded separately (see `MedSAM2/checkpoints/README.md`)
- Users will need to set up their own data directories

