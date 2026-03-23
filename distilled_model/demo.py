"""
PlacementDETR Demo — predict plausible object placements on background images.

Downloads checkpoints from HuggingFace and runs inference on bundled example images.

Usage:
    pip install -r requirements.txt
    python demo.py                        # uses bundled demo images
    python demo.py --image bg.jpg --class-name bottle --top-k 3
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── reproducibility ──────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from model import (
    PlacementDETR, MultiScaleBackbone, center_crop,
    IMG_SIZE, IMG_MEAN, IMG_STD,
)

# ── HuggingFace checkpoint download ─────────────────────────────────────────

HF_REPO = "marco-schouten/hidden-objects"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
CKPT_FILENAME = "placement_detr_ho.pth"


def download_checkpoint():
    """Download checkpoint from HuggingFace if not present locally."""
    ckpt_path = CKPT_DIR / CKPT_FILENAME
    if ckpt_path.exists():
        return ckpt_path
    print(f"Downloading checkpoint from huggingface.co/datasets/{HF_REPO} ...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"checkpoints/{CKPT_FILENAME}",
        repo_type="dataset",
        local_dir=CKPT_DIR.parent,
    )
    return Path(path)


# ── model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    model = PlacementDETR(
        num_classes=len(classes),
        num_queries=ckpt["model"]["query_offsets"].shape[0],
        use_cached_features=True,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    backbone = MultiScaleBackbone().to(device).eval()
    return model, classes, class_to_idx, backbone


# ── inference ────────────────────────────────────────────────────────────────

def predict(model, backbone, image, class_idx, device, top_k=5):
    """Predict top-k plausible bounding boxes for a class on a background image.

    Returns list of dicts with 'bbox' ([x,y,w,h] normalized) and 'score'.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    img = center_crop(image, IMG_SIZE)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = backbone(img_tensor)
        cls_tensor = torch.tensor([class_idx], device=device)
        pred_boxes, pred_plaus = model(feats, cls_tensor)

    scores = torch.sigmoid(pred_plaus[0]).cpu().numpy()
    boxes = pred_boxes[0].cpu().numpy()
    sorted_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in sorted_idx:
        x, y, w, h = boxes[idx]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(1.0, x + w), min(1.0, y + h)
        results.append({
            "bbox": [round(float(v), 4) for v in [x1, y1, x2 - x1, y2 - y1]],
            "score": round(float(scores[idx]), 4),
        })
    return results


# ── visualization ────────────────────────────────────────────────────────────

def visualize(image, predictions, output_path, class_name=None):
    """Draw predicted bboxes on the image and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img = center_crop(image, IMG_SIZE)
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img)

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(predictions), 1)))
    for i, pred in enumerate(predictions):
        x, y, w, h = pred["bbox"]
        rect = patches.Rectangle(
            (x * IMG_SIZE, y * IMG_SIZE), w * IMG_SIZE, h * IMG_SIZE,
            linewidth=2, edgecolor=colors[i], facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x * IMG_SIZE, y * IMG_SIZE - 4,
            f"#{i+1} ({pred['score']:.2f})",
            color=colors[i], fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    title = "PlacementDETR"
    if class_name:
        title += f" — \"{class_name}\""
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── demo examples ────────────────────────────────────────────────────────────

DEMO_IMAGES_DIR = Path(__file__).resolve().parent / "demo_input"

DEMO_EXAMPLES = [
    ("barn.jpg",           "horse"),
    ("bedroom.jpg",        "cat"),
    ("courtyard.jpg",      "bench"),
    ("desert_road.jpg",    "motorcycle"),
    ("dining_room.jpg",    "cake"),
    ("forest_path.jpg",    "bicycle"),
    ("office.jpg",         "laptop"),
    ("residential.jpg",    "car"),
]


def run_demo(model, backbone, classes, class_to_idx, device, top_k, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nModel classes ({len(classes)}): {classes}\n")
    print(f"Running demo on {len(DEMO_EXAMPLES)} images (top-{top_k} predictions)...\n")

    for rel_path, class_name in DEMO_EXAMPLES:
        img_path = DEMO_IMAGES_DIR / rel_path
        if not img_path.exists():
            print(f"  [skip] {rel_path} — not found")
            continue
        if class_name not in class_to_idx:
            print(f"  [skip] {rel_path} — class '{class_name}' not in model")
            continue

        image = Image.open(img_path).convert("RGB")
        preds = predict(model, backbone, image, class_to_idx[class_name], device, top_k)

        stem = f"{Path(rel_path).stem}_{class_name}"
        out_path = output_dir / f"{stem}.png"
        visualize(image, preds, out_path, class_name=class_name)

        print(f"  {rel_path}  class=\"{class_name}\"")
        for i, p in enumerate(preds):
            print(f"    #{i+1}  bbox={p['bbox']}  score={p['score']:.4f}")
        print(f"    → {out_path}\n")

    print(f"Done. Outputs saved to {output_dir}/")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PlacementDETR Demo")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (downloads from HF if omitted)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    parser.add_argument("--output-dir", default="demo_output", help="Directory for output images")

    # Single-image mode
    parser.add_argument("--image", help="Path to a single background image")
    parser.add_argument("--class-name", help="Object class name (e.g. 'bottle')")
    args = parser.parse_args()

    # Load model
    ckpt_path = args.checkpoint or download_checkpoint()
    device = torch.device(args.device)
    model, classes, class_to_idx, backbone = load_model(ckpt_path, device)

    # Single-image mode
    if args.image:
        if not args.class_name:
            parser.error("--class-name required with --image")
        if args.class_name not in class_to_idx:
            parser.error(f"Unknown class '{args.class_name}'. Available: {classes}")

        image = Image.open(args.image).convert("RGB")
        preds = predict(model, backbone, image, class_to_idx[args.class_name], device, args.top_k)

        print(f"\nTop-{args.top_k} predictions for '{args.class_name}':")
        for i, p in enumerate(preds):
            print(f"  #{i+1}  bbox={p['bbox']}  score={p['score']:.4f}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(args.image).stem}_{args.class_name}.png"
        visualize(image, preds, out_path, class_name=args.class_name)
        print(f"Saved → {out_path}")
        return

    # Demo mode (bundled examples)
    run_demo(model, backbone, classes, class_to_idx, device, args.top_k, args.output_dir)


if __name__ == "__main__":
    main()
