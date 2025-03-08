# OCR Pipeline with Llama-3.2-11B-Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

This repository contains an Optical Character Recognition (OCR) pipeline that leverages the Llama-3.2-11B-Vision model for text extraction from images. The pipeline includes both a baseline implementation using the pre-trained model and a fine-tuned version with significantly improved performance.

The project demonstrates substantial improvements in OCR accuracy through fine-tuning, making it suitable for various document understanding tasks.


## Models

This project uses the following models:

* **Base Model**: `unsloth/Llama-3.2-11B-Vision` - Used as the foundation for OCR tasks
* **Fine-tuned Model**: Custom fine-tuned version of Llama-3.2-11B-Vision optimized for OCR performance

## Features

* Image preprocessing for optimal OCR performance
* Text extraction using state-of-the-art vision-language models
* Fine-tuning capabilities to dramatically improve OCR accuracy
* Comprehensive evaluation metrics including WER, CER, Edit Distance, BLEU Score, and Exact Match Accuracy
* Support for batch processing of images

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ocr-pipeline.git
cd ocr-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration
import torch

# Load model and processor
model = MllamaForConditionalGeneration.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("unsloth/Llama-3.2-11B-Vision")

# Load an image
image = Image.open("path/to/your/image.jpg").convert("RGB")

# Create prompt
prompt = "<|image|><|begin_of_text|>Extract the text from this image."

# Process image and extract text
inputs = processor(image, prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024)
extracted_text = processor.decode(output[0]).strip()
print(extracted_text)
```

### Fine-tuning

The repository includes code for fine-tuning the Llama-3.2-11B-Vision model on custom OCR datasets:

```python
from unsloth import FastVisionModel

# Load model with Unsloth for efficient fine-tuning
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Configure LoRA parameters
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    r=16,
    lora_alpha=16,
)
```

For a complete fine-tuning example, see `final_submission.ipynb`

## Performance Comparison

The fine-tuned model shows dramatic improvements over the base model:

| Metric | Base Model | Fine-Tuned Model |
|--------|------------|------------------|
| WER | 1.1465 | 0.1041 |
| CER | 0.9231 | 0.0942 |
| Edit Distance | 3325.07 | 267.926 |
| BLEU Score | 0.3453 | 0.8848 |

These metrics demonstrate that the fine-tuned model achieves:
* ~91% reduction in Word Error Rate
* ~90% reduction in Character Error Rate
* ~92% reduction in Edit Distance
* ~156% improvement in BLEU Score

## Evaluation

The pipeline includes comprehensive evaluation metrics:
* Word Error Rate (WER)
* Character Error Rate (CER)
* Edit Distance
* BLEU Score
* Exact Match Accuracy


## Requirements

* Python 3.8+
* PyTorch 2.0+
* transformers
* unsloth
* Pillow
* pandas
* tqdm
* jiwer
* editdistance
* nltk

## Acknowledgements

* Unsloth for providing efficient fine-tuning tools
* Meta AI for developing the Llama 3.2 Vision model
