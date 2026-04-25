# train.py

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_lr_from_hr(hr_dir, lr_dir, scale=4):
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    lr_dir.mkdir(parents=True, exist_ok=True)

    for img_path in hr_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape[:2]
        lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(lr_dir / img_path.name), lr)


def copy_images(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for f in src_dir.rglob("*"):
        if f.suffix.lower() in IMG_EXTS:
            shutil.copy2(f, dst_dir / f.name)


# -------------------------
# SR Dataset
# -------------------------

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.files = sorted([p.name for p in self.hr_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        hr = cv2.imread(str(self.hr_dir / fname), cv2.IMREAD_GRAYSCALE)
        lr = cv2.imread(str(self.lr_dir / fname), cv2.IMREAD_GRAYSCALE)

        if hr is None or lr is None:
            raise FileNotFoundError(f"Cannot read image pair: {fname}")

        hr = torch.tensor(hr / 255.0).float().unsqueeze(0)
        lr = torch.tensor(lr / 255.0).float().unsqueeze(0)

        return lr, hr


# -------------------------
# Lightweight ESRGAN-style SR Model
# -------------------------

class SRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# -------------------------
# Super-resolution training
# -------------------------

def train_sr(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hr_dir = Path(args.root) / "data" / "part3" / "HR"
    lr_dir = Path(args.root) / "data" / "part3" / "LR"

    dataset = SRDataset(hr_dir, lr_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = SRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    save_dir = Path(args.root) / "checkpoints" / "sr"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for lr_img, hr_img in loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"[SR] Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.5f}")

        torch.save(model.state_dict(), save_dir / "last.pt")

    torch.save(model.state_dict(), save_dir / "best.pt")
    print(f"SR model saved to: {save_dir / 'best.pt'}")


# -------------------------
# Generate SR images
# -------------------------

@torch.no_grad()
def generate_sr_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.root) / "checkpoints" / "sr" / "best.pt"
    model = SRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    lr_dir = Path(args.root) / "data" / "part2" / "LR"
    sr_dir = Path(args.root) / "data" / "part2" / "SR"
    sr_dir.mkdir(parents=True, exist_ok=True)

    for img_path in lr_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        x = torch.tensor(img / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
        sr = model(x).squeeze().cpu().numpy()
        sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite(str(sr_dir / img_path.name), sr)

    print(f"Generated SR images saved to: {sr_dir}")


# -------------------------
# Classification
# -------------------------

def extract_feature(img):
    img = cv2.resize(img, (64, 64))
    return img.flatten() / 255.0


def load_cls_data(img_dir, labels_csv):
    img_dir = Path(img_dir)
    df = pd.read_csv(labels_csv)

    X, y = [], []

    for _, row in df.iterrows():
        img_path = img_dir / row["filename"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        X.append(extract_feature(img))
        y.append(row["label"])

    return np.array(X), np.array(y)


def train_classifier(args):
    img_dir = Path(args.root) / "data" / "part2" / args.cls_input
    labels_csv = Path(args.root) / "data" / "part2" / "labels.csv"

    X, y = load_cls_data(img_dir, labels_csv)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    pred = clf.predict(X)

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")

    print(f"[Classifier - {args.cls_input}] Accuracy: {acc * 100:.2f}%")
    print(f"[Classifier - {args.cls_input}] Macro-F1: {f1 * 100:.2f}%")

    results_dir = Path(args.root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "filename": pd.read_csv(labels_csv)["filename"][:len(pred)],
        "pred": pred
    }).to_csv(results_dir / f"predictions_{args.cls_input}.csv", index=False)


# -------------------------
# Class-level SISA
# -------------------------

CLASS_GROUPS = {
    "shard_00": ["canine", "central incisor", "lateral incisor"],
    "shard_01": ["first premolar", "second premolar"],
    "shard_02": ["first molar", "second molar", "third molar"],
}


def build_class_shards(args):
    labels_csv = Path(args.root) / "data" / "part2" / "labels.csv"
    sr_dir = Path(args.root) / "data" / "part2" / "SR"
    shard_root = Path(args.root) / "data" / "part2" / "class_shards"

    df = pd.read_csv(labels_csv)

    for shard_name, class_list in CLASS_GROUPS.items():
        shard_dir = shard_root / shard_name
        image_dir = shard_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        subset = df[df["label"].isin(class_list)].copy()
        subset.to_csv(shard_dir / "labels.csv", index=False)

        for _, row in subset.iterrows():
            src = sr_dir / row["filename"]
            if src.exists():
                shutil.copy2(src, image_dir / row["filename"])

    print(f"Class shards saved to: {shard_root}")


def unlearn_class(args):
    target_class = args.target_class
    shard_root = Path(args.root) / "data" / "part2" / "class_shards"

    affected = []

    for shard_name, class_list in CLASS_GROUPS.items():
        if target_class in class_list:
            affected.append(shard_name)

    if not affected:
        raise ValueError(f"Class '{target_class}' not found in CLASS_GROUPS.")

    for shard_name in affected:
        shard_dir = shard_root / shard_name
        labels_path = shard_dir / "labels.csv"

        df = pd.read_csv(labels_path)
        removed = df[df["label"] == target_class]
        remain = df[df["label"] != target_class]

        for _, row in removed.iterrows():
            img_path = shard_dir / "images" / row["filename"]
            if img_path.exists():
                img_path.unlink()

        remain.to_csv(labels_path, index=False)

        print(f"Unlearned class '{target_class}' from {shard_name}")
        print(f"Remaining samples: {len(remain)}")


# -------------------------
# Data preparation
# -------------------------

def prepare_data(args):
    root = Path(args.root)

    part1_hr = root / "data" / "part1" / "HR"
    part2_hr = root / "data" / "part2" / "HR"
    part3_hr = root / "data" / "part3" / "HR"

    part1_lr = root / "data" / "part1" / "LR"
    part2_lr = root / "data" / "part2" / "LR"
    part3_lr = root / "data" / "part3" / "LR"

    copy_images(args.part1_dir, part1_hr)
    copy_images(args.part2_dir, part2_hr)
    copy_images(args.part3_dir, part3_hr)

    create_lr_from_hr(part1_hr, part1_lr, args.scale)
    create_lr_from_hr(part2_hr, part2_lr, args.scale)
    create_lr_from_hr(part3_hr, part3_lr, args.scale)

    if args.labels_csv:
        dst = root / "data" / "part2" / "labels.csv"
        shutil.copy2(args.labels_csv, dst)
        print(f"Copied labels to: {dst}")

    print("Data preparation completed.")


# -------------------------
# Main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Dental SR + Class-level SISA")

    parser.add_argument("--mode", type=str, required=True,
                        choices=[
                            "prepare",
                            "train_sr",
                            "generate_sr",
                            "train_cls",
                            "build_shards",
                            "unlearn_class"
                        ])

    parser.add_argument("--root", type=str, default="dental_pipeline")

    parser.add_argument("--part1_dir", type=str, default=None)
    parser.add_argument("--part2_dir", type=str, default=None)
    parser.add_argument("--part3_dir", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None)

    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--cls_input", type=str, default="SR",
                        choices=["LR", "SR", "HR"])

    parser.add_argument("--target_class", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    Path(args.root).mkdir(parents=True, exist_ok=True)

    if args.mode == "prepare":
        prepare_data(args)

    elif args.mode == "train_sr":
        train_sr(args)

    elif args.mode == "generate_sr":
        generate_sr_images(args)

    elif args.mode == "train_cls":
        train_classifier(args)

    elif args.mode == "build_shards":
        build_class_shards(args)

    elif args.mode == "unlearn_class":
        if args.target_class is None:
            raise ValueError("--target_class is required for unlearn_class mode")
        unlearn_class(args)


if __name__ == "__main__":
    main()
