import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# PatchCore imports from the official repo
from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit
import patchcore.common
import patchcore.patchcore
import patchcore.utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global cache (populated at startup)
MODELS = {}          # { "bottle": PatchCore instance, ... }
MODELS_ROOT = None   # Path to folder containing mvtec_* dirs


# ---------- helpers ----------

def _find_models_root(model_dir: str) -> Path:
    """
    Try hard to find the directory that directly contains mvtec_* folders.
    Works for:
      - model_dir == experiment folder that contains 'models/'
      - model_dir == the 'models/' folder itself
      - model_dir with nested 'models/' somewhere below.
    """
    root = Path(model_dir)
    if not root.exists():
        raise RuntimeError(f"model_dir does not exist: {model_dir}")

    # Case 1: this directory itself looks like a models dir
    if any(d.is_dir() and d.name.startswith("mvtec_") for d in root.iterdir()):
        return root

    # Case 2: direct child 'models' folder
    direct = root / "models"
    if direct.exists():
        return direct

    # Case 3: search recursively for a 'models' folder that has mvtec_* subdirs
    for p in root.rglob("models"):
        if any(d.is_dir() and d.name.startswith("mvtec_") for d in p.iterdir()):
            return p

    raise RuntimeError(f"Could not find 'models' directory under {model_dir}")



def _load_patchcore_for_class(models_root: Path, class_name: str):
    """
    Load PatchCore for one class (used by model_fn).
    """
    cls_dir = models_root / f"mvtec_{class_name}"
    if not cls_dir.exists():
        raise ValueError(f"Model directory not found for class '{class_name}': {cls_dir}")

    nn_method = patchcore.common.FaissNN(
        on_gpu=False,      # <--- force CPU FAISS
        num_workers=4,
    )


    model = patchcore.patchcore.PatchCore(DEVICE)
    model.load_from_path(
        load_path=str(cls_dir),
        device=DEVICE,
        nn_method=nn_method,
    )
    model.eval()
    return model


def _write_image_to_mvtec_layout(pil_img: Image.Image, class_name: str) -> Path:
    """
    Save uploaded image in MVTec-like layout:
      /tmp/mvtec_infer/<class_name>/test/good/input.png
    """
    root = Path("/tmp/mvtec_infer")
    img_dir = root / class_name / "test" / "good"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / "input.png"
    pil_img.save(img_path)

    return root


def _create_dataloader(root: Path, class_name: str):
    """
    Create a single-image MVTec DataLoader (same transforms as official repo),
    but using the temporary /tmp/mvtec_infer tree we just wrote.
    """
    dataset = MVTecDataset(
        source=str(root),      # <-- /tmp/mvtec_infer
        classname=class_name,         # <-- just identify the dataset type, not a path
        resize=256,
        imagesize=224,
        split=DatasetSplit.TEST,
        seed=0,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    dataloader.name = f"mvtec_{class_name}"
    return dataloader




# ---------- SageMaker entrypoints ----------

def model_fn(model_dir):
    """
    Called once at container startup.

    Here we:
      1. Find the 'models' folder inside the untarred artifact.
      2. Preload PatchCore models for all 15 MVTec classes.
    """
    global MODELS_ROOT, MODELS

    MODELS_ROOT = _find_models_root(model_dir)

    # list of mvtec_* folders -> derive class names
    class_names = []
    for d in MODELS_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("mvtec_"):
            class_name = d.name.replace("mvtec_", "")
            class_names.append(class_name)

    # Preload all class models once (can take a minute but only at startup)
    for cls in class_names:
        MODELS[cls] = _load_patchcore_for_class(MODELS_ROOT, cls)

    print(f"[model_fn] Loaded PatchCore models for classes: {class_names}")

    return {
        "models_root": MODELS_ROOT,
        "device": DEVICE,
        "classes": class_names,
    }


def input_fn(request_body, content_type):
    """
    Expect JSON:
      {
        "class_name": "bottle",
        "image_base64": "<base64 string>"
      }
    """
    if content_type != "application/json":
        raise ValueError("Only application/json supported")

    data = json.loads(request_body)
    class_name = data["class_name"]
    img_bytes = base64.b64decode(data["image_base64"])
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    return {
        "class_name": class_name,
        "pil_img": pil_img,
    }


def predict_fn(inputs, context):
    """
    Light-weight per-request step:
      - write image to /tmp
      - build 1-image dataloader
      - run PatchCore.predict
    """
    class_name = inputs["class_name"]
    pil_img = inputs["pil_img"]

    if class_name not in MODELS:
        raise ValueError(f"Class '{class_name}' not in loaded models: {list(MODELS.keys())}")

    model = MODELS[class_name]

    # 1) Save image under fake MVTec root
    dataset_root = _write_image_to_mvtec_layout(pil_img, class_name)

    # 2) MVTec dataloader
    dataloader = _create_dataloader(dataset_root, class_name)

    # 3) Run predict
    with torch.no_grad():
        scores, segmentations, labels_gt, masks_gt = model.predict(dataloader)

    score = float(np.asarray(scores).reshape(-1)[0])

    return {
        "class_name": class_name,
        "anomaly_score": score,
    }


def output_fn(prediction, accept):
    return json.dumps(prediction)
