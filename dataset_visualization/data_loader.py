"""HiddenObjects dataset loader.

Loads annotations from HuggingFace and background images from a local
Places365 directory.  All images are resized and center-cropped to 512x512
to match the annotation coordinate space.

Usage:
    from data_loader import HiddenObjectsDataset, get_streaming_loader

    # Map-style (full download)
    ds = HiddenObjectsDataset("./data/places365", split="train")
    sample = ds[0]

    # Streaming (no full download)
    loader = get_streaming_loader("./data/places365", batch_size=32)
"""

import os

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class HiddenObjectsDataset(Dataset):
    """Map-style dataset that pairs HuggingFace annotations with local Places365 images."""

    _HF_REPO = "marco-schouten/hidden-objects"
    _IMG_SIZE = 512

    def __init__(self, places_root: str, split: str = "train", transform=None):
        self.data = load_dataset(self._HF_REPO, split=split)
        self.places_root = places_root
        self.transform = transform or T.Compose(
            [
                T.Resize(self._IMG_SIZE),
                T.CenterCrop(self._IMG_SIZE),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(os.path.join(self.places_root, item["bg_path"])).convert("RGB")
        img = self.transform(img)
        bbox = torch.tensor(item["bbox"], dtype=torch.float32) * self._IMG_SIZE
        return {
            "image": img,
            "bbox": bbox,
            "label": item["label"],
            "fg_class": item["fg_class"],
            "image_reward_score": item["image_reward_score"],
            "confidence": item["confidence"],
        }


def get_streaming_loader(places_root: str, split: str = "train", batch_size: int = 32):
    """Returns a DataLoader that streams from HuggingFace (no full download)."""
    repo = HiddenObjectsDataset._HF_REPO
    img_size = HiddenObjectsDataset._IMG_SIZE
    hf_dataset = load_dataset(repo, split=split, streaming=True)
    preprocess = T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor()])

    def collate_fn(batch):
        images, bboxes, labels, classes = [], [], [], []
        for item in batch:
            path = os.path.join(places_root, item["bg_path"])
            try:
                img = Image.open(path).convert("RGB")
            except FileNotFoundError:
                continue
            images.append(preprocess(img))
            bboxes.append(torch.tensor(item["bbox"], dtype=torch.float32) * img_size)
            labels.append(item["label"])
            classes.append(item["fg_class"])
        return {
            "image": torch.stack(images),
            "bbox": torch.stack(bboxes),
            "label": torch.tensor(labels),
            "fg_class": classes,
        }

    return DataLoader(hf_dataset, batch_size=batch_size, collate_fn=collate_fn)


if __name__ == "__main__":
    ds = load_dataset("marco-schouten/hidden-objects", streaming=True)
    row = next(iter(ds["train"]))
    print(row)
