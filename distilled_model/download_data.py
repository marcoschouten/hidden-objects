"""Download training data from HuggingFace.

Downloads the JSONL annotation files for PlacementDETR training from
https://huggingface.co/datasets/marco-schouten/hidden-objects

Usage:
    python download_data.py [--output_dir data]
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "marco-schouten/hidden-objects"
HF_REPO_TYPE = "dataset"

FILES = [
    "ho_irany_train_28_classes.jsonl",
    "ho_irany_test_28_classes.jsonl",
]


def download_data(output_dir: str = "data") -> dict[str, Path]:
    """Download JSONL files from HuggingFace and return local paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename in FILES:
        dest = out / filename
        if dest.exists():
            print(f"Already exists: {dest}")
            paths[filename] = dest
            continue
        print(f"Downloading {filename} ...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type=HF_REPO_TYPE,
            local_dir=str(out),
        )
        paths[filename] = dest
        print(f"  Saved to {dest}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HiddenObjects training data")
    parser.add_argument("--output_dir", default="data", help="Output directory (default: data)")
    args = parser.parse_args()
    download_data(args.output_dir)
