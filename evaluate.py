# evaluate.py

import argparse
from pathlib import Path

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import accuracy_score, f1_score


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def to_lpips_tensor(img, device):
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    img = img.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


@torch.no_grad()
def calc_lpips(hr, sr, lpips_fn, device):
    hr_tensor = to_lpips_tensor(hr, device)
    sr_tensor = to_lpips_tensor(sr, device)
    return float(lpips_fn(hr_tensor, sr_tensor).item())


def calc_edge_similarity(hr, sr):
    edge_hr = cv2.Canny(hr, 50, 150)
    edge_sr = cv2.Canny(sr, 50, 150)
    return structural_similarity(edge_hr, edge_sr, data_range=255)


def evaluate_sr(hr_dir, sr_dir, output_csv=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    hr_dir = Path(hr_dir)
    sr_dir = Path(sr_dir)

    results = []

    for hr_path in sorted(hr_dir.iterdir()):
        if hr_path.suffix.lower() not in IMG_EXTS:
            continue

        sr_path = sr_dir / hr_path.name
        if not sr_path.exists():
            print(f"Warning: SR image not found: {sr_path}")
            continue

        hr = read_gray(hr_path)
        sr = read_gray(sr_path)

        if sr.shape != hr.shape:
            sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

        psnr = peak_signal_noise_ratio(hr, sr, data_range=255)
        ssim = structural_similarity(hr, sr, data_range=255)
        edge_sim = calc_edge_similarity(hr, sr)
        lpips_score = calc_lpips(hr, sr, lpips_fn, device)

        results.append({
            "filename": hr_path.name,
            "PSNR": psnr,
            "SSIM": ssim,
            "EdgeSimilarity": edge_sim,
            "LPIPS": lpips_score
        })

    df = pd.DataFrame(results)

    if len(df) == 0:
        raise ValueError("No valid HR-SR image pairs found.")

    summary = {
        "PSNR_mean": df["PSNR"].mean(),
        "SSIM_mean": df["SSIM"].mean(),
        "EdgeSimilarity_mean": df["EdgeSimilarity"].mean(),
        "LPIPS_mean": df["LPIPS"].mean(),
        "num_images": len(df)
    }

    print("\n=== Super-Resolution Evaluation ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved detailed results to: {output_csv}")

    return df, summary


def evaluate_classification(labels_csv, predictions_csv):
    labels = pd.read_csv(labels_csv)
    preds = pd.read_csv(predictions_csv)

    merged = labels.merge(preds, on="filename", how="inner")

    if "label" not in merged.columns or "pred" not in merged.columns:
        raise ValueError("labels_csv must contain 'label'; predictions_csv must contain 'pred'.")

    acc = accuracy_score(merged["label"], merged["pred"])
    f1 = f1_score(merged["label"], merged["pred"], average="macro")

    print("\n=== Classification Evaluation ===")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro-F1: {f1 * 100:.2f}%")
    print(f"Samples: {len(merged)}")

    return {
        "Accuracy": acc,
        "Macro_F1": f1,
        "num_samples": len(merged)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluation script for Dental SR + SISA paper")

    parser.add_argument("--hr_dir", type=str, help="Path to high-resolution reference images")
    parser.add_argument("--sr_dir", type=str, help="Path to super-resolved images")
    parser.add_argument("--output_csv", type=str, default="sr_evaluation_results.csv")

    parser.add_argument("--labels_csv", type=str, default=None, help="Ground-truth labels CSV")
    parser.add_argument("--predictions_csv", type=str, default=None, help="Predictions CSV")

    args = parser.parse_args()

    if args.hr_dir and args.sr_dir:
        evaluate_sr(args.hr_dir, args.sr_dir, args.output_csv)

    if args.labels_csv and args.predictions_csv:
        evaluate_classification(args.labels_csv, args.predictions_csv)


if __name__ == "__main__":
    main()