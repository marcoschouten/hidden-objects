"""
PlacementDETR inference — predict plausible object placements on a background image.

Usage:
  # Single image, single class
  python inference.py --checkpoint checkpoints/placement_detr_ho.pth \
                      --image path/to/background.jpg \
                      --class-name "bottle" \
                      --top-k 5

  # Batch inference on a JSONL dataset
  python inference.py --checkpoint checkpoints/placement_detr_ho.pth \
                      --jsonl ../release_files/test_top1_iou.jsonl \
                      --places365-dir /path/to/Places365 \
                      --output predictions.json

  # Visualize predictions
  python inference.py --checkpoint checkpoints/placement_detr_ho.pth \
                      --image path/to/background.jpg \
                      --class-name "cat" \
                      --visualize --output viz.png
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import (
    PlacementDETR, MultiScaleBackbone, center_crop,
    IMG_SIZE, IMG_MEAN, IMG_STD, NUM_QUERIES,
)


def load_model(checkpoint_path, device="cuda"):
    """Load a PlacementDETR checkpoint.

    Returns:
        model: PlacementDETR in eval mode
        classes: list of class names
        class_to_idx: dict mapping class name → index
        backbone: MultiScaleBackbone in eval mode
    """
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


def predict(model, backbone, image, class_idx, device, top_k=5):
    """Run inference on a single PIL image.

    Args:
        model: PlacementDETR
        backbone: MultiScaleBackbone
        image: PIL Image (will be center-cropped to 512x512)
        class_idx: integer class index
        device: torch device
        top_k: number of top predictions to return

    Returns:
        list of dicts, each with:
          bbox: [x, y, w, h] normalized corner format in [0, 1]
          score: float plausibility score in [0, 1]
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

    # Sort by plausibility score
    sorted_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in sorted_idx:
        bbox = boxes[idx].tolist()
        # Clamp to [0, 1]
        x, y, w, h = bbox
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(1.0, x + w)
        y2 = min(1.0, y + h)
        bbox_clamped = [x1, y1, x2 - x1, y2 - y1]

        results.append({
            "bbox": [round(v, 6) for v in bbox_clamped],
            "score": round(float(scores[idx]), 6),
        })

    return results


def visualize(image, predictions, output_path, class_name=None):
    """Draw predicted bboxes on the image and save."""
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
            linewidth=2, edgecolor=colors[i], facecolor='none',
        )
        ax.add_patch(rect)
        label = f"#{i+1} ({pred['score']:.2f})"
        ax.text(x * IMG_SIZE, y * IMG_SIZE - 4, label,
                color=colors[i], fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    title = f"PlacementDETR predictions"
    if class_name:
        title += f" — {class_name}"
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PlacementDETR inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Single-image mode
    parser.add_argument("--image", help="Path to a background image")
    parser.add_argument("--class-name", help="Object class name (e.g. 'bottle')")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    parser.add_argument("--visualize", action="store_true", help="Save a visualization")

    # Batch mode
    parser.add_argument("--jsonl", help="Path to a JSONL dataset file for batch inference")
    parser.add_argument("--places365-dir", help="Root of Places365 images")

    parser.add_argument("--output", help="Output path (JSON for batch, PNG for visualize)")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, classes, class_to_idx, backbone = load_model(args.checkpoint, device)
    print(f"Loaded model with {len(classes)} classes: {classes}")

    # ── Single-image mode ────────────────────────────────────────────────
    if args.image:
        if not args.class_name:
            parser.error("--class-name is required with --image")
        if args.class_name not in class_to_idx:
            parser.error(f"Unknown class '{args.class_name}'. Available: {classes}")

        image = Image.open(args.image).convert("RGB")
        preds = predict(model, backbone, image, class_to_idx[args.class_name], device, args.top_k)

        print(f"\nTop-{args.top_k} predictions for '{args.class_name}':")
        for i, p in enumerate(preds):
            print(f"  #{i+1}  bbox={p['bbox']}  score={p['score']:.4f}")

        if args.visualize:
            out = args.output or "prediction.png"
            visualize(image, preds, out, class_name=args.class_name)

        if args.output and not args.visualize:
            with open(args.output, "w") as f:
                json.dump(preds, f, indent=2)
            print(f"Saved predictions → {args.output}")

    # ── Batch mode ───────────────────────────────────────────────────────
    elif args.jsonl:
        from tqdm import tqdm

        places_dir = args.places365_dir or os.environ.get("HO_PLACES365_DIR", "")
        if not places_dir:
            parser.error("--places365-dir is required for batch mode")

        # Read unique (entry_id, fg_class) pairs from the JSONL
        entries = {}  # entry_id → {bg_path, fg_class}
        with open(args.jsonl) as f:
            for line in f:
                row = json.loads(line)
                eid = row["entry_id"]
                if eid not in entries:
                    entries[eid] = {
                        "bg_path": os.path.join(places_dir, row["bg_path"]),
                        "fg_class": row["fg_class"],
                    }

        print(f"Running inference on {len(entries)} unique backgrounds...")
        all_preds = []
        skipped = 0
        for eid, meta in tqdm(sorted(entries.items())):
            cls = meta["fg_class"]
            if cls not in class_to_idx:
                skipped += 1
                continue
            if not os.path.isfile(meta["bg_path"]):
                skipped += 1
                continue

            image = Image.open(meta["bg_path"]).convert("RGB")
            preds = predict(model, backbone, image, class_to_idx[cls], device, args.top_k)
            all_preds.append({
                "entry_id": eid,
                "fg_class": cls,
                "predictions": preds,
            })

        if skipped:
            print(f"Skipped {skipped} entries (class not in model or image missing)")

        out = args.output or "batch_predictions.json"
        with open(out, "w") as f:
            json.dump(all_preds, f, indent=2)
        print(f"Saved {len(all_preds)} predictions → {out}")

    else:
        parser.error("Provide --image (single) or --jsonl (batch)")


if __name__ == "__main__":
    main()
