Bangla Fake News Detection — One-Page System Summary

Overview

A compact multimodal system that detects fake news in Bangla by combining text and image information. The pipeline ingests CSV rows (image_id, headline, label), preprocesses text and images, encodes each modality, fuses learned features, and classifies as Real vs Fake. Training supports efficient techniques (layer-freezing, AMP, gradient accumulation) and checkpoints. Inference supports batch CSV processing and real-time API calls.

Main flow

1. Data Ingestion
   - Input: CSV with columns (image_id, headline, label). Images stored under `images/`.
   - CSV may be a single file split into train/val/test (script supports splitting).

2. Preprocessing
   - Text: Tokenize with `csebuetnlp/banglabert`, pad/truncate to 128 tokens.
   - Image: Resize to 224×224, normalize with ImageNet mean/std.
   - Missing images: fallback to zero-tensor with a logged warning.

3. Model Architecture (forward pass)
   - Text encoder: pretrained Bangla transformer → last_hidden_state → Bi-LSTM → Attention pooling → text vector (512).
   - Image encoder: ResNet50 backbone → pooled feature → linear projection → image vector (512).
   - Fusion: Concatenate text and image vectors → MLP fusion layers → classifier → logits → softmax → probabilities.

4. Training
   - Loss: CrossEntropyLoss. Optimizer: AdamW with weight decay.
   - Scheduler: linear warmup + decay. Gradient clipping applied.
   - Efficiency: optional AMP (mixed precision), gradient accumulation, layer freezing for large models.
   - Checkpointing: best model saved by validation accuracy. Early stopping enabled.

5. Evaluation & Monitoring
   - Metrics: accuracy, precision, recall, F1, ROC-AUC, average precision.
   - Visualizations: confusion matrix, ROC/PR curves saved to `runs/`.

6. Inference
   - Batch mode: `inference.py` uses `BatchInference.predict_batch` to produce per-row outputs and save a CSV of predictions.
   - Real-time mode: `RealTimeInference.predict` returns JSON-like responses with prediction, confidence, probabilities and inference time.

Inputs / Outputs

- Inputs: CSV rows with `image_id`, `headline`, and (optional for inference) `label`. Images placed under `images/` with common extensions.
- Outputs: model checkpoint files (`models/`), inference CSV (`predictions.csv`) with columns [id, text, image_path, prediction, confidence, probabilities].

Operational notes

- Recommended Python: 3.8+. Use the repository venv (`.venv`).
- For GPU training: install a CUDA-compatible PyTorch; enable `--mixed_precision` to reduce memory.
- Reproducibility: save `requirements-lock.txt` (pip freeze) and the used `config.yaml` alongside checkpoints.

Printing & PDF generation

To generate a printable PDF from this one-page summary, install Pandoc and a TeX engine (e.g., MiKTeX) and run the PowerShell helper script `generate_summary_pdf.ps1` in the project folder. The script will produce `BanglaFakeNews_Summary.pdf`.

End of summary
