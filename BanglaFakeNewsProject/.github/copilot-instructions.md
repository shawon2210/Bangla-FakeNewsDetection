# Copilot Instructions - Bangla Fake News Detection

## Project Overview
Research-grade multimodal fake news detection system combining **BanglaBERT** (text) and **ResNet50** (images) with ensemble prediction. Three trained models average probabilities for robust classification.

## Architecture Essentials

### Model Design Philosophy
- **Two model architectures** coexist: `OptimizedMultimodalModel` (cross-attention fusion) and `MultimodalFakeNewsDetector` (LSTM+attention)
- **Ensemble predictor** (`predictor.py`) loads all 3 checkpoints and averages softmax outputs
- **Partial layer freezing** (BERT layers 0-5, ResNet layers 0-3) for efficient fine-tuning
- Images missing or corrupted? Fallback to zero tensors—no crashes

### Data Pipeline
- **Two dataset classes**: `data_loader.py` (config-based, production) vs `preprocess_data.py` (simplified, experiments)
- **Image loading**: Tries extensions `.png`, `.jpg`, `.jpeg`, `.jfif`, `.webp`, `.gif` automatically
- **Column flexibility**: Training scripts accept `--text_col`, `--image_col`, `--label_col` for different CSV schemas
- Image folder path: `images/` (configurable via `config.yaml` or hardcoded in `preprocess_data.py`)

## Key Workflows

### Training from Scratch
```powershell
# Default: uses merged_data.csv, trains MultimodalFakeNewsDetector
python train_multimodal.py

# Custom CSV with different column names
python train_multimodal.py --csv_path=custom.csv --text_col=text --image_col=img_id

# Resume from checkpoint (set env var before running)
$env:PRETRAINED_CHECKPOINT="outputs/checkpoints/checkpoint_epoch_5.pth"
python train_multimodal.py
```

### Inference & Deployment
```powershell
# Single prediction via CLI
python test_predictor.py

# Batch inference on CSV
python inference.py --data_path=Test.csv --output_path=predictions.csv

# Launch Gradio web UI
python app.py
```

### Python Environment Commands
Always use the virtual environment Python executable:
```powershell
C:/Users/Shawon/OneDrive/文档/BanglaFakeNewsProject/.venv/Scripts/python.exe <script>.py
```

## Project-Specific Patterns

### Device Selection
- `predictor.py` uses `_select_device()` helper that respects `config.yaml` `hardware.device` (`auto`/`cuda`/`cpu`)
- Training scripts default to `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

### Image Handling
- `predictor._resolve_image_path()` probes multiple extensions before giving up
- `preprocess_data.preprocess_image()` returns `None` on failure; caller creates zero tensor
- **Never raise on missing images**—models trained with blank images as fallback

### Model Loading Quirks
- Checkpoint dicts may contain `'model_state_dict'` key—extract before `load_state_dict()`
- `predictor.py` remaps old keys (`bert.` → `text_encoder.model.`, `resnet.` → `image_encoder.backbone.`) for backward compatibility
- Use `strict=False` when loading partial checkpoints to avoid key mismatch errors

### Training Script Conventions
- Early stopping tracked via `EarlyStopping` class in `train_multimodal.py` (patience=5 default)
- Checkpoints saved every `N` epochs (default 5) in `outputs/checkpoints/`
- Best model auto-saved as `best_multimodal_model.pth` or `multimodal_detector_v1.pth` based on script
- Always log to TensorBoard: `runs/` directory

## Configuration (`config.yaml`)
- **Frozen layers**: `freeze_text_layers: 6`, `freeze_image_layers: 4` (tune if overfitting/underfitting)
- **Learning rates**: `text_lr: 2e-5` (BERT), `image_lr: 1e-4` (ResNet), `other_lr: 1e-4` (classifier heads)
- **Early stopping**: `patience: 7`, `min_delta: 0.001` (increase patience if training unstable)
- **Image size**: `224` (ResNet standard; don't change without retraining)

## Common Pitfalls
1. **Path issues**: Windows paths with Unicode (文档) need raw strings or forward slashes
2. **CSV column names**: `headline` vs `text` vs `news_text`—always check with `df.columns` first
3. **CUDA OOM**: Reduce `batch_size` to 8 or 4; enable gradient accumulation in training loop
4. **Image IDs without extensions**: Dataset CSVs store base names (`politics_fake_236`), not full paths—code appends extensions

## Testing & Validation
- **Unit test**: `test_predictor.py` validates ensemble loading and text-only prediction
- **Image test**: Run predictor with known `.jfif` file (e.g., `images/politics_fake_236.jfif`)
- **Training smoke test**: Run 1-2 epochs on small subset to verify pipeline before full training

## References
- Text model: `csebuetnlp/banglabert` (Hugging Face)
- Image model: `torchvision.models.resnet50` (pretrained on ImageNet)
- Framework: PyTorch 2.8.0, Transformers 4.57.1, Gradio 5.49.1
