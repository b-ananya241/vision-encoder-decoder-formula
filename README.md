# Formula Recognition using Vision Transformer and RoBERTa

## Overview

This project implements an end-to-end deep learning system for converting images of mathematical formulas into structured textual representations (LaTeX-style). The model is based on a Vision Encoder-Decoder architecture that combines a Vision Transformer (ViT) encoder with a RoBERTa-based decoder.

## Architecture

* Encoder: Vision Transformer (ViT)
* Decoder: RoBERTa
* Tokenizer: Custom Byte Pair Encoding (BPE)

## Key Features

* Custom tokenizer trained on mathematical formula vocabulary
* Decoder pretraining using Masked Language Modeling (MLM)
* Multimodal learning using both image and text data
* Training on a combination of handwritten and synthetic datasets

## Dataset

The model is trained on:

* Handwritten mathematical formula dataset
* Synthetic formula dataset

*Note: Datasets are not included in this repository.*

## Training Pipeline

1. Train a BPE tokenizer on formula text
2. Pretrain the RoBERTa decoder using MLM
3. Train the VisionEncoderDecoder model (ViT + RoBERTa)
4. Fine-tune on combined datasets

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py <data_dir>
```

### Run inference

```bash
python inference.py <data_dir>
```

## Results

The model is capable of generating structured representations of mathematical formulas from images, handling both handwritten and synthetic inputs.

## Future Work

* Improve performance on complex expressions
* Incorporate evaluation metrics (e.g., BLEU, edit distance)
* Optimize training efficiency

## Report

Detailed methodology and experimental analysis are available in the `report/` directory.

## Author

* Ananya B
