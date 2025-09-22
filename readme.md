# Line-Level Guided Vulnerability Detection and Classification

This repository contains code and scripts for experiments investigating how line-level modifiability signals improve software vulnerability detection and classification. 

The experiments are based on three widely-used benchmark datasets: **Devign**, **Big-Vul** and **PrimeVul**.

This work is presented in **LOSVER: Line-Level Modifiability Signal-Guided Vulnerability Detection and Classification**, published at ASE 2025.

---

## UnixCoder Model Setup

HuggingFace does not fully support UnixCoder, so you must download it manually:

```bash
python download_unixcoder.py
```

Make sure the downloaded model (e.g., `unixcoder-nine/`) is placed in the appropriate location and used as `--model_name_or_path`.

---

## Dataset Preparation

### 1. Download Datasets

You must **manually download** the following datasets:

- [Devign Dataset]
- [Big-Vul Dataset]
- [PrimeVul Dataset(paired)]

### 2. Preprocessing the Datasets

After downloading, place the datasets in the correct directories:

- Put the **Devign** dataset in `detection/`
- Put the **Big-Vul** dataset in `classification/`
- Put the **PrimeVul** dataset in `detection_primevul/`

Then, run the following scripts to preprocess the raw datasets:


```bash
# Before preprocess Devign, clone the following repositories into `detection/`:
git clone https://github.com/FFmpeg/FFmpeg.git detection/FFmpeg

git clone https://github.com/qemu/qemu.git detection/qemu

# Then run:
cd detection
python devign_preprocess.py
```

```bash
# Preprocess Big-Vul
cd classification
python bigvul_preprocess.py
```

```bash
# Preprocess PrimeVul
cd detection_primevul
python primevul_preprocess.py
```
---

## Checkpoints for Quick Testing

To test without retraining, download the pretrained checkpoints from the following anonymous Figshare link: [Download Checkpoints (Figshare)](https://figshare.com/s/5beb0f16a9233f73852b?file=54953453)

---

## Running Experiments

Below are the core commands to reproduce the experiments.

To train the Weighted Vulnerability Detector/Classifier, include the `--localized_location` argument and set it to the directory of the corresponding localizer.

### Detection (Devign)

#### 1. Modifiable Line Localizer

```bash
python run_line.py \
  --output_dir=./unix_512_localizer \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --train_data_file=train_unix_512.jsonl \
  --eval_data_file=val_unix_512.jsonl \
  --test_data_file=test_unix_512.jsonl \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

#### 2. Weighted Vulnerability Detector

```bash
python run_weighted.py \
  --output_dir=./unix_512_detector \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --localized_location=./unix_512_localizer \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

#### 3. Baseline Comparison

For CodeBERT and UnixCoder:

```bash
python run_base.py \
  --output_dir=./unix_512_base \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --train_data_file=train_unix_512.jsonl \
  --eval_data_file=val_unix_512.jsonl \
  --test_data_file=test_unix_512.jsonl \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

For CodeT5+:

```bash
python run_base_codeT5p.py \
  --output_dir=./codet5p_512_base \
  --model_type codet5 \
  --model_name_or_path=Salesforce/codet5p-220m \
  --tokenizer_name=Salesforce/codet5p-220m \
  --train_data_file=train_codet5p.jsonl \
  --eval_data_file=val_codet5p.jsonl \
  --test_data_file=test_codet5p.jsonl \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

---

### Classification (Big-Vul CWE)

#### 1. Modifiable Line Localizer

```bash
python run_line_CWE.py \
  --output_dir=./unix_512_localizer_CWE \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --train_data_file=CWE_train_unix_512.jsonl \
  --eval_data_file=CWE_val_unix_512.jsonl \
  --test_data_file=CWE_test_unix_512.jsonl \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

#### 2. Weighted Vulnerability Classifier

```bash
python run_weighted_CWE.py \
  --output_dir=./unix_512_classifier \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --localized_location=./unix_512_localizer_CWE \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

#### 3. Baseline Comparison (Unified Script for All PLMs)

```bash
python run_base_CWE.py \
  --output_dir=./unix_512_base_CWE \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --train_data_file=CWE_train_unix_512.jsonl \
  --eval_data_file=CWE_val_unix_512.jsonl \
  --test_data_file=CWE_test_unix_512.jsonl \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

---
### Detection (PrimeVul)

#### 1. Modifiable Line Localizer

```bash
python run_line_primevul.py \
  --output_dir=./unix_512_primevul_localizer \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --train_data_file=primevul_train_paired_gt.jsonl \
  --eval_data_file=primevul_valid_paired_gt.jsonl \
  --test_data_file=primevul_test_paired_gt.jsonl \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

#### 2. Weighted Vulnerability Detector

```bash
python run_weighted_primevul.py \
  --output_dir=./unix_512_primevul_detector \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --localized_location=./unix_512_primevul_localizer \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test \
  --use_ground_truth
```
Add ```use_ground_truth``` to utilize ground-truth lines, exclude it to utilize predicted lines.

#### 3. Baseline Comparison

```bash
python run_base_primevul.py \
  --output_dir=./unix_512_primevul_base \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --train_data_file=primevul_train_paired_gt.jsonl \
  --eval_data_file=primevul_valid_paired_gt.jsonl \
  --test_data_file=primevul_test_paired_gt.jsonl \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

---

## Ablation Study (RQ2)

To run the ablation study setup (used for RQ2 in the paper), use:

```bash
python run_ablation_study.py \
  --output_dir=./unix_512_ablation \
  --model_type roberta \
  --model_name_or_path=../unixcoder-nine \
  --tokenizer_name=../unixcoder-nine \
  --localized_location=./unix_512_localizer \
  --block_size=512 \
  --seed=123456 \
  --do_train \
  --do_test
```

---

## GPT-4o Evaluation

To evaluate GPT-4o's ability to utilize line-level focus signals, use the run_gpt.py script. 

### Instructions

1. Open the file `run_gpt.py`.
2. Fill in the following fields inside the script:
   - `api_key`: your OpenAI API key.
   - `input_file`: the path to the output directory of the Modifiable Line Localizer (e.g., `./unix_512_localizer/`).
3. Run the script using:

```bash
python run_gpt.py
```

---

## Repository Structure

```
├── classification/
│   ├── bigvul_preprocess.py
│   ├── run_base_CWE.py
│   ├── run_line_CWE.py
│   └── run_weighted_CWE.py
├── detection/
│   ├── devign_preprocess.py
│   ├── run_ablation_study.py
│   ├── run_base.py
│   ├── run_base_codeT5p.py
│   ├── run_gpt.py
│   ├── run_line.py
│   └── run_weighted.py
├── detection_primevul/
│   ├── primevul_preprocess.py
│   ├── run_base_primevul.py
│   ├── run_line_primevul.py
│   └── run_weighted_primevul.py
├── download_unixcoder.py
└── README.md
```

---

## License

This code is for research purposes. The datasets (Devign, Big-Vul, PrimeVul) and UnixCoder model must be used in accordance with their respective licenses and terms of use.
