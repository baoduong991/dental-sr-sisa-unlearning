# Dental X-ray Super-Resolution & SISA Unlearning

This repository provides the implementation of our research on combining **image super-resolution** and **class-level machine unlearning** for panoramic dental radiographs.

---

## Overview

Panoramic dental X-ray images often suffer from limited resolution, which can affect both clinical interpretation and automated analysis. At the same time, modern AI systems need the ability to **adapt when certain categories of data must be removed**.

This project addresses both challenges through a unified pipeline:

- **Super-Resolution (SR)** to enhance image quality  
- **Classification** to evaluate downstream utility  
- **SISA-based Unlearning** to efficiently remove class-specific information

---

## Pipeline
High-Resolution Images (Subset 3)
↓
Super-Resolution Model
↓
Enhanced Images (SR Output)
↓
Classification Model
↓
SISA-based Unlearning
↓
Updated Model (after class removal)

## Datasets:
We use the **Mendeley Panoramic Dental X-ray Dataset (Version 3)**:
- **Subset 3** → Super-resolution training  
- **Subset 2** → Classification & unlearning  
- **Subset 1** → Structural validation
  
Website to the [original dataset](https://data.mendeley.com/datasets/73n3kz2k4k/3) (Accessed on 25 April 2026)

## 📊 Evaluation Metrics

### Image Quality
- PSNR  
- SSIM  
- LPIPS  
- Edge Similarity  

### Classification
- Accuracy  
- F1-score  

### Unlearning
- Retraining time  
- Performance retention  

---

## Results

| Task | Metric | Result |
|------|--------|--------|
| Super-Resolution | PSNR | ~29.6 dB |
| Classification (LR) | Accuracy | ~72% |
| Classification (SR) | Accuracy | ~79% |
| Unlearning (SISA) | Retraining Time | ~34% |

---

## Installation
<pre><code>
git clone https://github.com/your-username/dental-sr-sisa-unlearning.git
cd dental-sr-sisa-unlearning
pip install -r requirements.txt
</code></pre>

## Usage
Run the following command:
<pre><code>
python evaluate.py \
  --hr_dir data/part1/HR \
  --sr_dir data/part1/SR \
  --output_csv results/part1_sr_metrics.csv
</code></pre>

<pre><code>
  python evaluate.py \
  --labels_csv data/part2/labels.csv \
  --predictions_csv results/predictions.csv
</code></pre>


![Python](https://img.shields.io/badge/python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

