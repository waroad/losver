# Line-Level Guided Vulnerability Detection and Classification

This repository contains code and scripts for experiments investigating how line-level modifiability signals improve software vulnerability detection and classification. The experiments are based on two widely-used benchmark datasets: **Devign** and **Big-Vul**.

---

## 📦 Dataset Preparation

### 1. Download Datasets

You must **manually download** the following datasets:

- [Devign Dataset]
- [Big-Vul Dataset]

### 2. Filtering the Datasets

After downloading, place the datasets in the correct directories:

- Put the **Devign** dataset in `detection/`
- Put the **Big-Vul** dataset in `classification/`

Then, run the following scripts to filter the raw datasets:

```bash
# Filter Big-Vul
python classification/CWE_filtering.py
```

```bash
# Before filtering Devign, clone the following repositories into `detection/`:
git clone https://github.com/FFmpeg/FFmpeg.git detection/FFmpeg

git clone https://github.com/qemu/qemu.git detection/qemu

# Then run:
python detection/devign_filtering.py
```

---

## 📥 UnixCoder Model Setup

HuggingFace does not fully support UnixCoder, so you must download it manually:

```bash
python download_unixcoder.py
```

Make sure the downloaded model (e.g., `unixcoder-nine/`) is placed in the appropriate location and used as `--model_name_or_path`.

---

## 🔧 Running Experiments

Below are the core commands to reproduce the experiments.

### 🔍 Detection (Devign)

#### 1. Modifiable Line Localizer

```bash
python run_line.py \
  --output_dir=./unix1 \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --block_size=1024 \
  seed=123456
```

#### 2. Weighted Vulnerability Detection

```bash
python run_weighted.py \
  --output_dir=./unix1 \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --block_size=1024 \
  seed=123456
```

#### 3. Baseline Comparison

For CodeBERT and UnixCoder:

```bash
python run_base.py \\
  --output_dir=./unix1 \\
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --block_size=1024 \
  seed=123456
```

For CodeT5+:

```bash
python run_base_codeT5p.py \
  --output_dir=./unix1 \
  --model_type roberta \
  --model_name_or_path=Salesforce/codet5p-220m \
  --tokenizer_name=Salesforce/codet5p-220m \
  --block_size=1024 \
  seed=123456
```

---

### 🧠 Classification (Big-Vul CWE)

#### 1. Modifiable Line Localizer

```bash
python run_line_CWE.py \
  --output_dir=./unix1 \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --block_size=1024 \
  seed=123456
```

#### 2. Weighted Vulnerability Classification

```bash
python run_weighted_CWE.py \
  --output_dir=./unix1 \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --block_size=1024 \
  seed=123456
```

#### 3. Baseline Comparison (Unified Script for All PLMs)

```bash
python run_base_CWE.py \
  --output_dir=./unix1 \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --block_size=1024 \
  seed=123456
```

---

## 🧪 Ablation Study (RQ2)

To run the ablation study setup (used for RQ2 in the paper), use:

```bash
python run_ablation_study.py \
  --output_dir=./unix1 \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --block_size=1024 \
  seed=123456
```

---

## 🗂️ Repository Structure

```
├── classification/
│   ├── CWE_filtering.py
│   ├── run_line_CWE.py
│   ├── run_weighted_CWE.py
│   └── run_base_CWE.py
├── detection/
│   ├── devign_filtering.py
│   ├── run_line.py
│   ├── run_weighted.py
│   ├── run_base.py
│   ├── run_base_codeT5p.py
│   └── run_ablation_study.py
├── download_unixcoder.py
├── README.md
```

---

## 📜 License

This code is for research purposes. The datasets (Devign, Big-Vul) and UnixCoder model must be used in accordance with their respective licenses and terms of use.
