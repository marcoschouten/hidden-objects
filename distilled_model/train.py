"""
Train PlacementDETR for object placement prediction.

Trains on Hidden Objects annotations (JSONL format) and evaluates on a held-out test set.
Data is automatically downloaded from HuggingFace if not present locally.

Usage:
    python train.py --places365_dir data/Places365 \
                    --filter_b 20 --min_confidence 0.7

    # Or with explicit JSONL paths:
    python train.py --train_jsonl data/ho_irany_train_28_classes.jsonl \
                    --test_jsonl data/ho_irany_test_28_classes.jsonl \
                    --places365_dir data/Places365

Data format (JSONL, one entry per line):
    {"bg_path": "relative/path.jpg", "fg_class": "bottle",
     "bbox": [x, y, w, h], "label": 1,
     "image_reward_score": 0.85, "confidence": 0.95}
"""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from model import (
    PlacementDETR,
    MultiScaleBackbone,
    center_crop,
    IMG_SIZE,
    IMG_MEAN,
    IMG_STD,
    NUM_QUERIES,
)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Dataset ──────────────────────────────────────────────────────────────────


class PlacementDataset(Dataset):
    """Groups JSONL entries by (bg_path, fg_class) with optional filtering."""

    def __init__(self, jsonl_path, class_to_idx, transform, places365_dir,
                 min_boxes_per_bg=None, top_k_boxes=None, min_confidence=None,
                 backbone=None, device=None, cache_dir=None):
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.places365_dir = Path(places365_dir)
        self.backbone = backbone
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Group by (bg_path, fg_class), positive labels only
        raw_groups = {}
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                if row["fg_class"] not in class_to_idx:
                    continue
                if row.get("label", 1) != 1:
                    continue
                if min_confidence is not None and row.get("confidence", 1.0) < min_confidence:
                    continue
                bg = row["bg_path"]
                if not bg.startswith("/"):
                    bg = str(self.places365_dir / bg)
                key = (bg, row["fg_class"])
                if key not in raw_groups:
                    raw_groups[key] = []
                raw_groups[key].append({
                    "bbox": row["bbox"],
                    "reward": row.get("image_reward_score", 1.0),
                })

        # Apply filters
        groups = {}
        for key, items in raw_groups.items():
            if min_boxes_per_bg and len(items) < min_boxes_per_bg:
                continue
            if top_k_boxes and len(items) > top_k_boxes:
                items = sorted(items, key=lambda x: x["reward"], reverse=True)[:top_k_boxes]
            groups[key] = {
                "boxes": [it["bbox"] for it in items],
                "rewards": [it["reward"] for it in items],
            }

        self.keys = list(groups.keys())
        self.groups = groups
        print(f"  Loaded {len(self.keys)} groups from {jsonl_path}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        bg_path, fg_class = self.keys[idx]
        group = self.groups[(bg_path, fg_class)]

        if self.cache_dir is not None and self.backbone is not None:
            cache_key = hashlib.md5(bg_path.encode()).hexdigest()[:12]
            cache_path = self.cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                data = torch.load(cache_path, weights_only=True)
                feat = {"c4": data["c4"], "c5": data["c5"]}
            else:
                img = Image.open(bg_path).convert("RGB")
                img = center_crop(img, IMG_SIZE)
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat_dict = self.backbone(img_tensor)
                    c4 = feat_dict["c4"].cpu().squeeze(0)
                    c5 = feat_dict["c5"].cpu().squeeze(0)
                torch.save({"c4": c4, "c5": c5}, cache_path)
                feat = {"c4": c4, "c5": c5}
        else:
            img = Image.open(bg_path).convert("RGB")
            img = center_crop(img, IMG_SIZE)
            feat = self.transform(img)

        class_idx = self.class_to_idx[fg_class]
        boxes = torch.tensor(group["boxes"], dtype=torch.float32).clamp(0, 1)
        rewards = torch.tensor(group["rewards"], dtype=torch.float32)
        return feat, class_idx, boxes, rewards


def collate_fn(batch):
    first_feat = batch[0][0]
    if isinstance(first_feat, dict):
        feats = {"c4": torch.stack([b[0]["c4"] for b in batch]),
                 "c5": torch.stack([b[0]["c5"] for b in batch])}
    else:
        feats = torch.stack([b[0] for b in batch])
    return (feats,
            torch.tensor([b[1] for b in batch]),
            [b[2] for b in batch],
            [b[3] for b in batch])


# ── Loss ─────────────────────────────────────────────────────────────────────


def compute_giou_matrix(preds, gt_boxes):
    """GIoU between all pred/gt pairs. Boxes in [x, y, w, h] format."""
    p, g = preds.unsqueeze(1), gt_boxes.unsqueeze(0)
    p_x1, p_y1 = p[..., 0], p[..., 1]
    p_x2, p_y2 = p_x1 + p[..., 2], p_y1 + p[..., 3]
    g_x1, g_y1 = g[..., 0], g[..., 1]
    g_x2, g_y2 = g_x1 + g[..., 2], g_y1 + g[..., 3]

    inter_x1 = torch.max(p_x1, g_x1)
    inter_y1 = torch.max(p_y1, g_y1)
    inter_x2 = torch.min(p_x2, g_x2)
    inter_y2 = torch.min(p_y2, g_y2)
    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    p_area = p[..., 2] * p[..., 3]
    g_area = g[..., 2] * g[..., 3]
    union = p_area + g_area - inter
    iou = inter / (union + 1e-8)

    enc_x1 = torch.min(p_x1, g_x1)
    enc_y1 = torch.min(p_y1, g_y1)
    enc_x2 = torch.max(p_x2, g_x2)
    enc_y2 = torch.max(p_y2, g_y2)
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
    return iou - (enc_area - union) / (enc_area + 1e-8)


def hungarian_loss(pred_boxes, pred_plausibility, gt_boxes_list, rewards_list, num_queries):
    """Multi-target Hungarian matching loss with reward weighting."""
    from scipy.optimize import linear_sum_assignment

    B = pred_boxes.shape[0]
    device = pred_boxes.device
    total_bbox, total_plaus, total_matched = 0.0, 0.0, 0

    for b in range(B):
        preds = pred_boxes[b]
        plaus = pred_plausibility[b]
        gt = gt_boxes_list[b].to(device)
        rewards = rewards_list[b].to(device)
        M = gt.shape[0]

        if M == 0:
            total_plaus += nn.functional.binary_cross_entropy_with_logits(
                plaus, torch.zeros_like(plaus))
            continue

        l1_cost = torch.abs(preds.unsqueeze(1) - gt.unsqueeze(0)).sum(dim=2)
        giou = compute_giou_matrix(preds, gt)
        cost = l1_cost + (1 - giou)

        pi, gi = linear_sum_assignment(cost.detach().cpu().numpy())

        bbox_loss = 0.0
        for p_idx, g_idx in zip(pi, gi):
            w = 0.5 + 0.5 * torch.clamp(rewards[g_idx], 0, 1)
            bbox_loss += w * (5.0 * l1_cost[p_idx, g_idx] + 2.0 * (1 - giou[p_idx, g_idx]))

        if len(pi) > 0:
            total_bbox += bbox_loss / len(pi)
            total_matched += 1

        max_iou, _ = giou.max(dim=1)
        total_plaus += nn.functional.mse_loss(
            torch.sigmoid(plaus), torch.clamp(max_iou, 0, 1))

    return total_bbox / max(total_matched, 1) + 0.5 * total_plaus / B


# ── Training ─────────────────────────────────────────────────────────────────


def _move(feats, device):
    if isinstance(feats, dict):
        return {k: v.to(device) for k, v in feats.items()}
    return feats.to(device)


def train(train_loader, val_loader, num_classes, device,
          epochs=100, lr=1e-4, patience=15, output_dir=None,
          classes=None, class_to_idx=None, args=None):
    model = PlacementDETR(
        num_classes=num_classes,
        num_queries=NUM_QUERIES,
        use_cached_features=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses, val_losses = [], []
    best_val, best_state, best_epoch = float("inf"), None, 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for feats, cls_idx, boxes, rewards in pbar:
            feats, cls_idx = _move(feats, device), cls_idx.to(device)
            pred_boxes, pred_plaus = model(feats, cls_idx)
            loss = hungarian_loss(pred_boxes, pred_plaus, boxes, rewards, NUM_QUERIES)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=loss.item())
        train_losses.append(total / len(train_loader))

        model.eval()
        vtotal = 0
        with torch.no_grad():
            for feats, cls_idx, boxes, rewards in val_loader:
                feats, cls_idx = _move(feats, device), cls_idx.to(device)
                loss = hungarian_loss(*model(feats, cls_idx), boxes, rewards, NUM_QUERIES)
                vtotal += loss.item()
        val_losses.append(vtotal / len(val_loader))

        print(f"Epoch {epoch+1}: train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

        if val_losses[-1] < best_val:
            best_val, best_epoch = val_losses[-1], epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  -> New best (val_loss={best_val:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Save checkpoint
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "model": best_state,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "best_epoch": best_epoch,
            "num_queries": NUM_QUERIES,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        ckpt_path = output_dir / "placement_detr_ho.pth"
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # Loss plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.axvline(x=best_epoch - 1, color="r", linestyle="--", label=f"Best (epoch {best_epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("PlacementDETR Training")
        plt.savefig(output_dir / "loss.png", dpi=150)
        plt.close()

    return model, train_losses, val_losses, best_epoch


# ── Evaluation ───────────────────────────────────────────────────────────────


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / (w1 * h1 + w2 * h2 - inter + 1e-8)


def evaluate(model, test_loader, device, class_to_idx):
    model.eval()
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    metrics = defaultdict(lambda: {"iou1": [], "iou5": [], "hit1": [], "hit5": []})

    with torch.no_grad():
        for feats, cls_idx, boxes_list, _ in tqdm(test_loader, desc="Evaluating"):
            feats = _move(feats, device)
            pred_boxes, pred_plaus = model(feats, cls_idx.to(device))
            scores = torch.sigmoid(pred_plaus)

            B = feats["c5"].shape[0] if isinstance(feats, dict) else feats.shape[0]
            for b in range(B):
                s = scores[b].cpu().numpy()
                bx = pred_boxes[b].cpu().numpy()
                order = np.argsort(-s)
                gt = boxes_list[b].numpy()
                cls = idx_to_class[cls_idx[b].item()]
                if len(gt) == 0:
                    continue

                iou1 = max(compute_iou(bx[order[0]], g) for g in gt)
                metrics[cls]["iou1"].append(iou1)
                metrics[cls]["hit1"].append(1 if iou1 >= 0.5 else 0)

                top5 = [max(compute_iou(bx[order[k]], g) for g in gt)
                        for k in range(min(5, len(order)))]
                best5 = max(top5)
                metrics[cls]["iou5"].append(best5)
                metrics[cls]["hit5"].append(1 if best5 >= 0.5 else 0)

    # Print results
    print(f"\n{'Class':<20} {'N':<6} {'IoU@1':<10} {'IoU50@1':<10} {'IoU@5':<10} {'IoU50@5':<10}")
    print("-" * 66)
    all_m = {"iou1": [], "iou5": [], "hit1": [], "hit5": []}
    for cls in sorted(metrics):
        m = metrics[cls]
        print(f"{cls:<20} {len(m['iou1']):<6} "
              f"{np.mean(m['iou1']):<10.4f} {np.mean(m['hit1']):<10.4f} "
              f"{np.mean(m['iou5']):<10.4f} {np.mean(m['hit5']):<10.4f}")
        for k in all_m:
            all_m[k].extend(m[k])
    print("-" * 66)
    print(f"{'OVERALL':<20} {len(all_m['iou1']):<6} "
          f"{np.mean(all_m['iou1']):<10.4f} {np.mean(all_m['hit1']):<10.4f} "
          f"{np.mean(all_m['iou5']):<10.4f} {np.mean(all_m['hit5']):<10.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train PlacementDETR")
    parser.add_argument("--train_jsonl", default=None,
                        help="Training JSONL file (auto-downloaded from HuggingFace if omitted)")
    parser.add_argument("--test_jsonl", default=None,
                        help="Test JSONL file (auto-downloaded from HuggingFace if omitted)")
    parser.add_argument("--places365_dir", required=True, help="Root of Places365 images")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache dir for ResNet features (speeds up training)")
    parser.add_argument("--filter_b", type=int, default=20,
                        help="Keep top-K boxes per background by reward (default: 20)")
    parser.add_argument("--min_confidence", type=float, default=0.7,
                        help="Min detection confidence to keep a bbox (default: 0.7)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    # Auto-download data from HuggingFace if paths not provided
    if args.train_jsonl is None or args.test_jsonl is None:
        from download_data import download_data
        data_dir = Path(__file__).parent / "data"
        paths = download_data(str(data_dir))
        if args.train_jsonl is None:
            args.train_jsonl = str(data_dir / "ho_irany_train_28_classes.jsonl")
        if args.test_jsonl is None:
            args.test_jsonl = str(data_dir / "ho_irany_test_28_classes.jsonl")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    backbone = MultiScaleBackbone().to(device).eval()

    # Discover classes
    classes = set()
    with open(args.train_jsonl) as f:
        for line in f:
            classes.add(json.loads(line)["fg_class"])
    classes = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    # Build dataset
    train_ds = PlacementDataset(
        args.train_jsonl, class_to_idx, transform, args.places365_dir,
        top_k_boxes=args.filter_b, min_confidence=args.min_confidence,
        backbone=backbone, device=device, cache_dir=args.cache_dir,
    )

    # Train/val split (90/10)
    val_size = max(1, int(len(train_ds) * 0.1))
    train_size = len(train_ds) - val_size
    train_split, val_split = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_split, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_split, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # Train
    model, *_ = train(
        train_loader, val_loader, len(classes), device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        output_dir=args.output_dir, classes=classes, class_to_idx=class_to_idx,
        args=args,
    )

    # Evaluate on test set
    if args.test_jsonl:
        test_ds = PlacementDataset(
            args.test_jsonl, class_to_idx, transform, args.places365_dir,
            backbone=backbone, device=device, cache_dir=args.cache_dir,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_fn)
        evaluate(model, test_loader, device, class_to_idx)


if __name__ == "__main__":
    main()
