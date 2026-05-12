# Bangla Fake News Detection

A multimodal deep learning system for detecting fake news in Bangla media. Combines **BanglaBERT/BART** (text) with **ResNet50** (images) using cross-modal attention, and ensembles multiple models for robust predictions. Ships with a **Gradio web UI** for interactive analysis.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

This project addresses the growing problem of misinformation in Bangla-language media by building a multimodal fake news detector that analyzes both **text content** (headlines and articles) and **associated images**. The system uses an ensemble of up to 4 models to produce a final prediction with a confidence score.

**How it works:**
1. A Bangla news headline + article is tokenized using a pretrained Bangla transformer
2. An associated news image (optional) is processed through a pretrained ResNet50
3. Text and image features are fused via cross-modal attention
4. Multiple models vote via probability averaging (ensemble)
5. The final output is **Real** or **Fake** with a confidence percentage

---

## Architecture

### Model v1 — `MultimodalFakeNewsDetector`
```
BanglaBERT → BiLSTM → Attention → Text Features (512-d)
                                                    ├─ Concat → FC → 2 classes
ResNet50  → FC        → Image Features (512-d)
```

### Model v2 — `OptimizedMultimodalModel` (primary)
```
BanglaBART Encoder (frozen first 6 layers)
  → Hidden States (512-d) ──┐
                             ├─ Cross-Modal Attention ──┐
ResNet50 Encoder (frozen first 4 layers)                │
  → Image Features (512-d)  ────────────────────────────┤
                             │                          │
                    [text_pooled | image_features | attended] → FC(512→256→2)
```

### Ensemble Strategy
The `Predictor` class loads all available model checkpoints and averages their softmax probabilities:
- **Model 1:** Primary Optimized (`outputs/best_multimodal_model.pth`)
- **Model 2:** Training Checkpoint (`outputs/checkpoints/checkpoint_epoch_5.pth`)
- **Model 3:** Multimodal V1 (`best_multimodal_model.pth`)
- **Model 4:** Enhanced (`outputs/enhanced_best_model.pth`)

Missing checkpoints are handled gracefully — the ensemble runs with whatever models are available.

---

## Key Features

- **Multimodal fusion** — combines text (BanglaBERT/BART) and image (ResNet50) features via cross-modal attention
- **4-model ensemble** — probability averaging across multiple trained checkpoints for higher accuracy
- **Gradio web UI** — interactive browser-based interface at `http://localhost:7860`
- **News type classification** — keyword-based categorization (Politics, Business, Crime, Sports, Entertainment, Health, Technology, General)
- **Confidence scoring** — outputs confidence level (Very High / High / Good / Moderate / Low)
- **Batch inference** — `inference.py` supports batch processing of CSV datasets
- **Comprehensive evaluation** — accuracy, precision, recall, F1, ROC-AUC, PR curves, confusion matrix
- **TensorBoard logging** — training metrics logged to `runs/`
- **Early stopping** — configurable patience with best-weight restoration
- **Mixed device support** — auto-detects CUDA, falls back to CPU

---

## Project Structure

```
BanglaFakeNewsProject/
├── app.py                          # Gradio web UI (port 7860)
├── predictor.py                    # Ensemble predictor (loads all available models)
├── config.yaml                     # All hyperparameters and paths
├── model_defs.py                   # Model architecture definitions
│   ├── MultimodalFakeNewsDetector  # v1: BERT + BiLSTM + Attention + ResNet50
│   ├── OptimizedMultimodalModel    # v2: BanglaBART + ResNet50 + CrossModalAttention
│   ├── BanglaBARTEncoder           # Text encoder with frozen layers
│   ├── EfficientResNet50           # Image encoder with frozen layers
│   ├── CrossModalAttention         # Cross-modal attention mechanism
│   ├── AttentionLayer              # Attention pooling for LSTM outputs
│   └── ModelManager                # Model save/load utility
├── preprocess_data.py              # Dataset preparation, tokenization, image transforms
├── data_loader.py                  # DataLoader creation (backward-compat wrapper)
├── train_multimodal.py             # Training script for v1 model
├── train_optimized.py              # Training script for v2 model (with TensorBoard, early stopping)
├── train_pipeline.py               # Unified training pipeline
├── inference.py                    # Batch inference on CSV datasets
├── evaluation.py                   # Comprehensive evaluation (ROC, PR, confusion matrix, F1)
├── evaluate_testset.py             # Test-set evaluation script
├── merge_and_prepare_data.py       # Merge and prepare CSV datasets
├── fix_merged_data.py              # Data cleaning utilities
├── verify_merged_data.py           # Data verification
├── quick_start.py                  # Quick start helper
├── test_predictor.py               # Predictor unit test
├── test_updated_predictor.py       # Updated predictor test
├── clean_gradio_cache.py           # Gradio cache cleanup
├── pipeline_integration.py         # Pipeline integration utilities
├── process_and_detect.py           # End-to-end detection script
├── style.css                       # Gradio UI custom styles
├── config-Shawon.yaml              # Alternate config (Shawon's setup)
├── Requirements.txt                # Python dependencies
├── Requirements-Shawon.txt         # Alternate requirements
├── QUICK_START.md                  # Quick start guide
├── Train.csv                       # Training data
├── Validation.csv                  # Validation data
├── Test.csv                        # Test data
├── Authentic-48K.csv               # Authentic news data
├── Fake-1K.csv                     # Fake news data
├── LabeledAuthentic-7K.csv         # Labeled authentic data
├── LabeledFake-1K.csv              # Labeled fake data
├── merged_data.csv                 # Merged dataset
├── detection_results.csv           # Output predictions
├── images/                         # News images (organized by category)
│   ├── business_fake_*.png
│   ├── business_real_*.webp
│   ├── crime_fake_*.webp
│   ├── crime_real_*.webp
│   ├── entertainment_fake_*.webp
│   ├── entertainment_real_*.webp
│   ├── politics_fake_*.webp
│   ├── politics_real_*.webp
│   ├── sports_fake_*.webp
│   ├── sports_real_*.webp
│   ├── technology_fake_*.webp
│   └── technology_real_*.webp
├── models/                         # Additional model scripts
│   ├── advanced_training.py
│   ├── enhanced_model.py
│   ├── enhanced_predictor.py
│   ├── enhanced_app.py
│   ├── unified_predictor.py
│   ├── training_monitor.py
│   └── ...
├── outputs/
│   ├── best_multimodal_model.pth   # Primary trained model
│   ├── best_multimodal_model_history.json
│   ├── enhanced_best_model.pth     # Enhanced model
│   ├── checkpoints/                # Training checkpoints
│   └── training_history.png        # Training curves plot
├── runs/                           # TensorBoard event logs
├── utils/
│   └── utils.py                    # Training history plotting utility
└── ReadMe/                         # Additional documentation
    ├── DETECTION_RESULTS.md
    ├── ENHANCED_README.md
    ├── ENHANCEMENT_SUMMARY.md
    ├── FIXES_APPLIED.md
    ├── MODELS_EXPLAINED.md
    ├── MODEL_STATUS.md
    ├── ONE_PAGE_SUMMARY.md
    ├── PROJECT_STATUS.md
    └── SYSTEM_DIAGNOSTIC_REPORT.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Git

### Setup

```bash
# Clone the repository
git clone git@github.com:shawon2210/Bangla-FakeNewsDetection.git
cd Bangla-FakeNewsDetection/BanglaFakeNewsProject

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/WSL
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# Install dependencies
pip install -r Requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| torchvision | >=0.15.0 | Image models (ResNet50) |
| transformers | >=4.30.0 | BanglaBERT/BART tokenizer & model |
| pandas | >=2.0.0 | Data loading and manipulation |
| numpy | >=1.24.0 | Numerical computation |
| scikit-learn | >=1.3.0 | Evaluation metrics |
| matplotlib | >=3.7.0 | Training plots |
| seaborn | >=0.12.0 | Visualization |
| tqdm | >=4.65.0 | Progress bars |
| gradio | >=4.0.0 | Web UI |
| pyyaml | >=6.0 | Configuration parsing |
| tensorboard | >=2.13.0 | Training monitoring |
| Pillow | >=10.0.0 | Image processing |

---

## Quick Start

### Launch the Web UI

```bash
cd BanglaFakeNewsProject
python app.py
```

The app opens at `http://localhost:7860`. If the port is busy, it automatically tries 7862–7865.

### Test the Predictor

```bash
python test_updated_predictor.py
```

Expected output:
```
Loading primary model...
Loading checkpoint model...
Loading MultimodalFakeNewsDetector V1...
Loading enhanced model...
✅ Loaded 4 models: Primary Optimized, Checkpoint, Multimodal V1, Enhanced
✅ SUCCESS: Using 4-model ensemble!
```

### Using the UI

1. Enter a Bangla news **headline** in the first text box
2. Enter the full **article** text in the second text box (at least 5 words)
3. Optionally upload a related **news image**
4. Click **"Analyze News"**
5. View the result: **Real** ✅ or **Fake** ❌ with confidence gauge and news type

---

## Training

### Train the Optimized Model (v2)

```bash
python train_optimized.py \
  --config config.yaml \
  --data-dir . \
  --output-dir outputs
```

### Train the Baseline Model (v1)

```bash
python train_multimodal.py
```

### Resume from Checkpoint

```bash
python train_optimized.py \
  --config config.yaml \
  --data-dir . \
  --resume outputs/checkpoints/checkpoint_epoch_5.pth
```

### Training Configuration (config.yaml)

```yaml
model:
  text_model_name: "csebuetnlp/banglabert"
  num_classes: 2
  text_feature_dim: 512
  image_feature_dim: 512
  fusion_dim: 256
  dropout_rate: 0.3
  freeze_text_layers: 6        # Freeze first 6 encoder layers
  freeze_image_layers: 4       # Freeze first 4 ResNet blocks

training:
  batch_size: 16
  epochs: 20
  text_lr: 1e-5               # Lower LR for pretrained text encoder
  image_lr: 1e-4              # Higher LR for image projection
  other_lr: 1e-4              # LR for classifier / fusion layers
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0

early_stopping:
  patience: 7
  min_delta: 0.001
  restore_best_weights: true

data:
  max_text_length: 128
  image_size: 224
  image_folder: "images"
  train_csv: "Train.csv"
  val_csv: "Validation.csv"
  test_csv: "Test.csv"
```

### Training Features
- **Layer freezing** — early layers of BanglaBERT and ResNet50 are frozen to reduce overfitting and speed up training
- **Differential learning rates** — lower LR for pretrained encoders, higher LR for new layers
- **Linear warmup** — 10% of training steps use a learning rate warmup
- **Gradient clipping** — max norm of 1.0 for training stability
- **Early stopping** — stops training if validation loss doesn't improve for 7 epochs
- **TensorBoard logging** — all metrics logged to `runs/` directory
- **Checkpoint saving** — saves every N epochs (configurable via `logging.save_every`)

---

## Inference

### Single Prediction (Python API)

```python
from predictor import Predictor

predictor = Predictor(config_path="config.yaml")
result = predictor.predict(
    text="ভূমিকম্পে কেঁপে উঠলো ঢাকা। আজ সকালে ঢাকা শহরে এক শক্তিশালী ভূমিকম্প অনুভূত হয়েছে।",
    image_path="images/politics_fake_1.png"
)

print(result)
# {
#   "prediction": "Fake",
#   "confidence": 0.87,
#   "probabilities": [0.13, 0.87],
#   "news_type": "General",
#   "model_count": 4
# }
```

### Batch Inference

```bash
python inference.py \
  --model_path outputs/best_multimodal_model.pth \
  --input_csv Test.csv \
  --output_csv predictions.csv \
  --batch_size 32
```

### Command-Line Arguments (inference.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `outputs/best_multimodal_model.pth` | Path to trained model |
| `--input_csv` | `Test.csv` | Input CSV with text and image columns |
| `--output_csv` | `predictions.csv` | Output CSV with predictions |
| `--batch_size` | 32 | Inference batch size |
| `--device` | `auto` | Device: auto, cuda, cpu |
| `--image_folder` | `images` | Folder containing news images |

---

## Evaluation

### Run Full Evaluation

```bash
python evaluate_testset.py
```

### Evaluation Metrics

The `evaluation.py` module computes and visualizes:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total |
| **Precision** | True positives / (true positives + false positives) |
| **Recall** | True positives / (true positives + false negatives) |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |
| **Average Precision** | Area under the Precision-Recall curve |
| **Confusion Matrix** | Visual breakdown of TP, FP, TN, FN |

### Generate Plots

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator(class_names=['Real', 'Fake'])
results = evaluator.evaluate_predictions(y_true, y_pred, y_prob)
evaluator.plot_confusion_matrix(y_true, y_pred)
evaluator.plot_roc_curve(y_true, y_prob)
evaluator.plot_pr_curve(y_true, y_prob)
```

---

## Configuration

All settings are centralized in `config.yaml`:

```yaml
# Model
model:
  text_model_name: "csebuetnlp/banglabert"   # Pretrained Bangla transformer
  num_classes: 2                               # Binary: Real / Fake
  text_feature_dim: 512
  image_feature_dim: 512
  fusion_dim: 256
  dropout_rate: 0.3
  freeze_text_layers: 6
  freeze_image_layers: 4

# Training
training:
  batch_size: 16
  epochs: 20
  text_lr: 1e-5
  image_lr: 1e-4
  other_lr: 1e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0

# Early Stopping
early_stopping:
  patience: 7
  min_delta: 0.001
  restore_best_weights: true

# Data
data:
  max_text_length: 128
  image_size: 224
  image_folder: "images"
  train_csv: "Train.csv"
  val_csv: "Validation.csv"
  test_csv: "Test.csv"

# Logging
logging:
  log_dir: "runs"
  checkpoint_dir: "checkpoints"
  save_every: 5
  model_save_path: "models/optimized_bangla_fake_news.pth"

# Hardware
hardware:
  device: "auto"    # auto-detect CUDA
  num_workers: 4
  pin_memory: true
  mixed_precision: false
```

---

## Model Details

### Text Encoder: BanglaBART
- **Base model:** `csebuetnlp/banglabert` (Bangla BERT from BUET)
- **Frozen layers:** First 6 encoder layers (of 12)
- **Output:** 512-dimensional text features
- **Tokenization:** WordPiece tokenizer, max length 128

### Image Encoder: EfficientResNet50
- **Base model:** ResNet50 pretrained on ImageNet-1K
- **Frozen layers:** First 4 residual blocks
- **Output:** 512-dimensional image features
- **Input:** 224×224 RGB images, ImageNet normalized

### Cross-Modal Attention
- Projects both text and image features into a shared 256-d space
- Uses 8-head multi-head attention
- Text queries attend to image keys/values
- Output is concatenated with raw text and image features for classification

### Classifier Head
```
[text_pooled (512) | image_features (512) | cross_attended (256)] → 1284-d
  → Linear(1284, 512) → ReLU → Dropout(0.3)
  → Linear(512, 256) → ReLU → Dropout(0.3)
  → Linear(256, 2) → Softmax
```

### Weight Initialization
- Classifier layers: Xavier uniform initialization, zero bias
- Pretrained encoders: Keep original pretrained weights

---

## Dataset

The project uses a labeled Bangla news dataset with both text and images:

| File | Description | Size |
|------|-------------|------|
| `Train.csv` | Training split | ~80% of data |
| `Validation.csv` | Validation split | ~10% of data |
| `Test.csv` | Test split | ~10% of data |
| `Authentic-48K.csv` | Authentic news articles | ~48K samples |
| `Fake-1K.csv` | Fake news articles | ~1K samples |
| `LabeledAuthentic-7K.csv` | Labeled authentic subset | ~7K samples |
| `LabeledFake-1K.csv` | Labeled fake subset | ~1K samples |
| `images/` | News images by category | ~1000+ images |

### Image Categories
Images are organized in `images/` with naming convention: `{category}_{label}_{id}.{ext}`
- **Categories:** business, crime, entertainment, politics, sports, technology
- **Labels:** fake, real
- **Formats:** .png, .jpg, .webp

### CSV Format
Each row contains:
- `headline` — Bangla news headline (text)
- `image_id` — filename of the associated image
- `label` — 0 (Real) or 1 (Fake)

---

## Troubleshooting

### "No models could be loaded"
Ensure model checkpoints exist at the paths specified in `config.yaml`. Run training first, or download pretrained weights.

### "Cannot find config.yaml"
Use absolute paths:
```bash
python train_optimized.py --config "/full/path/to/config.yaml"
```

### Port already in use
The app automatically tries ports 7860–7865. To manually specify:
```python
demo.launch(server_port=8080)
```

### CUDA out of memory
Reduce `batch_size` in `config.yaml` (try 8 or 4), or set `hardware.device` to `cpu`.

### Gradio 403 errors
Clear the Gradio cache:
```bash
rm -rf gradio_cached_examples/
python clean_gradio_cache.py
```

### Import errors
Make sure you're running from the `BanglaFakeNewsProject/` directory:
```bash
cd BanglaFakeNewsProject
python app.py
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+ |
| **NLP** | HuggingFace Transformers (BanglaBERT/BART) |
| **Computer Vision** | TorchVision (ResNet50) |
| **Web UI** | Gradio 4.0+ |
| **Data** | Pandas, NumPy |
| **Evaluation** | Scikit-learn, Matplotlib, Seaborn |
| **Monitoring** | TensorBoard |
| **Config** | YAML |

---

## License

This project is developed as part of a B.Sc. Computer Science & Engineering thesis at IUBAT (International University of Business Agriculture and Technology), Dhaka, Bangladesh.

**Author:** Md Shawon Molla
**GitHub:** [github.com/shawon2210](https://github.com/shawon2210)
**Email:** 22103330@iubat.edu
