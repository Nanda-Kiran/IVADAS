import os
import io
import json
import pickle

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from src import softpatch, common


# ------------ SageMaker hooks ------------ #

def model_fn(model_dir):
    """
    Load ALL mvtec_* class models from:
      model_dir/models/mvtec_bottle
      model_dir/models/mvtec_cable
      ...
    Returns a dict with:
      - models_by_class: { "bottle": [coreset_0, ...], ... }
      - device, preprocess
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_root = os.path.join(model_dir, "models")
    if not os.path.isdir(models_root):
        raise RuntimeError(f"models directory not found at {models_root}")

    # Discover all mvtec_* subfolders
    class_dirs = [
        d for d in os.listdir(models_root)
        if d.startswith("mvtec_") and os.path.isdir(os.path.join(models_root, d))
    ]
    if not class_dirs:
        raise RuntimeError(f"No mvtec_* class folders found in {models_root}")

    models_by_class = {}
    resize = None
    imagesize = None

    for d in class_dirs:
        full_path = os.path.join(models_root, d)
        mvtec_class = d.split("mvtec_")[1]  # e.g. "bottle"

        meta_path = os.path.join(full_path, "meta.pkl")
        if not os.path.isfile(meta_path):
            raise RuntimeError(f"Missing meta.pkl in {full_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        num_coresets = meta["num_coresets"]
        args = meta["args"]

        if resize is None:
            resize = args.get("resize", 256)
        if imagesize is None:
            imagesize = args.get("imagesize", 224)

        nn_method = common.FaissNN(
            on_gpu=False,
            num_workers=4,
            device=device.index if device.type == "cuda" else -1,
        )

        coresets = []
        for i in range(num_coresets):
            c = softpatch.SoftPatch(device)
            coreset_dir = os.path.join(full_path, f"coreset_{i}")
            c.load_from_path(coreset_dir, device=device, nn_method=nn_method)
            coresets.append(c)

        models_by_class[mvtec_class] = coresets

    # Shared preprocessing (MVTec / ImageNet style)
    preprocess = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return {
        "models_by_class": models_by_class,
        "device": device,
        "preprocess": preprocess,
    }


def input_fn(request_body, content_type):
    """
    Multi-class input.

    We require JSON:
      {
        "classname": "bottle",   # one of: bottle, cable, capsule, ...
        "image": "<base64 bytes>"
      }
    """
    if content_type != "application/json":
        raise ValueError(
            "For multi-class SoftPatch, please use content_type='application/json' "
            "with keys: 'classname' and 'image' (base64)."
        )

    body = json.loads(request_body)

    classname = body.get("classname")
    if classname is None:
        raise ValueError("Missing 'classname' in request JSON.")

    import base64
    img_bytes = base64.b64decode(body["image"])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    return {"classname": classname, "image": image}


def predict_fn(input_data, model):
    """
    Run SoftPatch for the specified MVTec subclass.
    """
    classname = input_data["classname"]
    pil_image = input_data["image"]

    models_by_class = model["models_by_class"]
    preprocess = model["preprocess"]
    device = model["device"]

    if classname not in models_by_class:
        raise ValueError(
            f"Unknown classname '{classname}'. "
            f"Available: {list(models_by_class.keys())}"
        )

    coresets = models_by_class[classname]

    #disable soft weights at inference (we didn't serialize coreset_weight)
    for coreset in coresets:
        coreset.soft_weight_flag = False

    x = preprocess(pil_image).unsqueeze(0).to(torch.float).to(device)  # [1, 3, H, W]

    all_scores = []
    all_masks = []

    for coreset in coresets:
        scores, masks = coreset._predict(x)
        all_scores.append(scores[0])
        all_masks.append(masks[0])

    all_scores = np.array(all_scores, dtype=np.float32)
    all_masks = np.stack(all_masks, axis=0)  # [N_coresets, H, W]

    anomaly_score = float(all_scores.mean())
    anomaly_heatmap = all_masks.mean(axis=0)

    return {
        "classname": classname,
        "anomaly_score": anomaly_score,
        "anomaly_heatmap": anomaly_heatmap.tolist(),
    }



def output_fn(prediction, accept):
    if accept.startswith("application/json"):
        return json.dumps(prediction), "application/json"
    return json.dumps(prediction), "application/json"
