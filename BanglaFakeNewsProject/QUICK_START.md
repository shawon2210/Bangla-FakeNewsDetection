# Quick Start Guide - Bangla Fake News Detection

## 🚀 Running the Application

### Launch the Gradio Web UI
```powershell
# Activate virtual environment (if not already active)
.venv\Scripts\Activate.ps1

# Navigate to project
cd BanglaFakeNewsProject

# Run the app
python app.py
```

The app will open at `http://localhost:7860` (or another port if 7860 is busy).

---

## 🧪 Testing

### Quick Predictor Test
```powershell
cd BanglaFakeNewsProject
python test_updated_predictor.py
```

Expected output:
```
✅ Loaded 4 models: Primary Optimized, Checkpoint, Multimodal V1, Enhanced
✅ SUCCESS: Using 4-model ensemble!
```

---

## 🏋️ Training

### Train from Scratch
```powershell
python BanglaFakeNewsProject/train_optimized.py `
  --config "BanglaFakeNewsProject/config.yaml" `
  --data-dir "BanglaFakeNewsProject" `
  --output-dir "outputs"
```

### Resume from Checkpoint
```powershell
python BanglaFakeNewsProject/train_optimized.py `
  --config "BanglaFakeNewsProject/config.yaml" `
  --data-dir "BanglaFakeNewsProject" `
  --resume "outputs/checkpoints/checkpoint_epoch_5.pth"
```

---

## 📊 Current Model Status

### Available Models (4-Model Ensemble)
✅ **Model 1:** Primary Optimized (`outputs/best_multimodal_model.pth`)  
✅ **Model 2:** Checkpoint Epoch 5 (`outputs/checkpoints/checkpoint_epoch_5.pth`)  
✅ **Model 3:** Multimodal V1 (`multimodal_detector_v1.pth`)  
✅ **Model 4:** Enhanced (`outputs/enhanced_best_model.pth`)

### Prediction UI Shows:
```
Models: 4 | Enhanced Analysis
```

---

## 🗂️ Project Structure

```
BanglaFakeNewsProject/
├── app.py                          # Gradio web interface
├── predictor.py                    # 4-model ensemble predictor
├── config.yaml                     # Configuration file
├── Train.csv, Validation.csv, Test.csv  # Datasets
├── images/                         # News images
├── outputs/
│   ├── best_multimodal_model.pth   # Primary trained model
│   ├── enhanced_best_model.pth     # Enhanced model
│   ├── checkpoints/                # Training checkpoints
│   └── runs/                       # TensorBoard logs
├── models/                         # Model implementations
└── utils/                          # Helper utilities
```

---

## 🧹 Cleanup (Optional)

### Remove Old TensorBoard Logs
```powershell
# Archive first (optional)
Compress-Archive -Path ".\runs" -DestinationPath ".\runs_backup.zip"

# Delete
Remove-Item -Recurse -Force ".\runs"
```

### Clean Outputs (will require retraining)
```powershell
# ⚠️ WARNING: This deletes all trained models!
Remove-Item -Recurse -Force ".\outputs"
```

---

## 🔧 Troubleshooting

### Issue: "Cannot find config.yaml"
**Solution:** Provide full path to config:
```powershell
python train_optimized.py --config "BanglaFakeNewsProject/config.yaml"
```

### Issue: "Cannot find Train.csv"
**Solution:** Specify data directory:
```powershell
python train_optimized.py --data-dir "BanglaFakeNewsProject"
```

### Issue: "Models: 1" shows in UI
**Solution:** Fixed! The updated predictor now loads all 4 models automatically.
Test with:
```powershell
python test_updated_predictor.py
```

### Issue: App doesn't start
**Solutions:**
1. Check if port is busy - app will try ports 7860-7865
2. Clear Gradio cache:
   ```powershell
   Remove-Item -Recurse -Force ".\gradio_cached_examples"
   ```

---

## 📝 Configuration

### Key Settings in `config.yaml`

```yaml
training:
  batch_size: 8
  epochs: 15
  text_lr: 2e-5
  image_lr: 1e-4

logging:
  save_every: 5          # Save checkpoint every N epochs
  model_save_path: "outputs/best_multimodal_model.pth"

hardware:
  device: "auto"         # auto, cuda, cpu
  num_workers: 4
```

---

## 🎯 Next Steps

1. **Run the app:** `python app.py`
2. **Test prediction:** Try example inputs in the UI
3. **Verify models:** Should show "Models: 4"
4. **Check confidence:** High confidence (>80%) indicates reliable predictions

---

## 📚 Additional Resources

- **Training Details:** See `FIXES_SUMMARY.md`
- **Model Architecture:** See `models/` folder
- **ReadMe Documentation:** See `ReadMe/` folder

---

**Last Updated:** November 1, 2025  
**Status:** ✅ All 4 models working in ensemble
