import os, tarfile
from pathlib import Path

# 1. Point to your experiment folder (the one in the screenshot)
exp_root = Path("MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_2")
models_dir = exp_root / "models"

assert models_dir.exists(), f"Models dir not found: {models_dir}"

# (Optional) sanity check a couple of classes
print("Subfolders:", [p.name for p in models_dir.iterdir() if p.is_dir()])

# 2. Create model.tar.gz that contains EVERYTHING under `models/`
tar_path = "patchcore_model.tar.gz"
if os.path.exists(tar_path):
    os.remove(tar_path)

with tarfile.open(tar_path, "w:gz") as tar:
    # this will create paths like "./models/mvtec_bottle/..."
    tar.add(models_dir, arcname="models")

print("Created:", tar_path)
