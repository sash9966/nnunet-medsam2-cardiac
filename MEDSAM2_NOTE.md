# Note on MedSAM2 Directory

The `MedSAM2/` directory is a separate git repository (cloned from the original MedSAM2 repo).

## Options for GitHub

### Option 1: Copy Only Modified File (Current Approach)
The modified inference script is saved as `medsam2_infer_3D_CT_modified.py` in the root directory.
Users can copy this to their MedSAM2 installation.

### Option 2: Add as Git Submodule
If you want to track the full MedSAM2 repository:

```bash
# Remove MedSAM2 from current repo
git rm -r --cached MedSAM2
rm -rf MedSAM2/.git

# Add as submodule
git submodule add https://github.com/bowang-lab/MedSAM2.git MedSAM2
git commit -m "Add MedSAM2 as submodule"

# Then apply your modifications
cp medsam2_infer_3D_CT_modified.py MedSAM2/medsam2_infer_3D_CT.py
git add MedSAM2/medsam2_infer_3D_CT.py
git commit -m "Apply modifications to MedSAM2 inference script"
```

### Option 3: Remove .git and Add All Files
If you want to include the full MedSAM2 code in your repo:

```bash
rm -rf MedSAM2/.git
git add MedSAM2/
git commit -m "Include MedSAM2 codebase"
```

**Note**: This may violate MedSAM2's license terms. Check their LICENSE file first.

## Recommended Approach

For now, we're using Option 1 (copy modified file only) to:
- Respect MedSAM2's repository structure
- Keep the repo lightweight
- Make it clear what modifications were made

Users should:
1. Clone MedSAM2 separately: `git clone https://github.com/bowang-lab/MedSAM2.git`
2. Copy `medsam2_infer_3D_CT_modified.py` to `MedSAM2/medsam2_infer_3D_CT.py`
3. Follow MedSAM2's installation instructions

