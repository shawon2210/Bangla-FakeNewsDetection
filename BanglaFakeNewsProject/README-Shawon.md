# Bangla Fake News Detection - Optimized Research Model

## Overview

This project implements an efficient multimodal model for Bangla fake news detection using Bangla BART and ResNet50. The system is optimized for real-world deployment and research applications, combining text and image features through advanced cross-modal attention mechanisms.

## Key Features

- **Optimized Architecture**: Efficient combination of Bangla BART and ResNet50
- **Cross-Modal Attention**: Advanced fusion of text and image features
- **Comprehensive Training**: Early stopping, checkpointing, and monitoring
- **Real-world Ready**: Batch and real-time inference capabilities
- **Research-Grade Evaluation**: Extensive metrics and visualization tools

## Architecture

### Model Components

1. **Text Encoder**: Bangla BART with frozen early layers for efficiency
2. **Image Encoder**: ResNet50 with optimized feature extraction
3. **Cross-Modal Attention**: Multi-head attention for text-image fusion
4. **Classifier**: Multi-layer neural network with dropout

### Key Optimizations

- Layer freezing for faster training
- Gradient clipping for stable training
- Different learning rates for different components
- Efficient feature projection and fusion

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd BanglaFakeNewsProject
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python test_device.py
```

## Data Preparation

### Data Format

Your CSV files should contain:
- `image_id`: Unique identifier for images
- `headline`: Bangla text content
- `label`: 0 (Real) or 1 (Fake)

### Directory Structure

```
project/
├── images/              # Image folder
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── Train.csv           # Training data
├── Validation.csv      # Validation data
├── Test.csv           # Test data
└── ...
```

## Usage

### 1. Training the Model

#### Quick Start
```bash
python train_optimized.py
```

#### Custom Configuration
Edit the configuration in `config.yaml` to customize your training session.

### 2. Running the Demo

To launch the web-based demo, run:

```bash
python app.py
```

This will start a Gradio interface where you can input text and an image to get a prediction.

### 3. Model Evaluation
...

#### Comprehensive Evaluation
```python
from evaluation import ModelEvaluator

# Load predictions and ground truth
evaluator = ModelEvaluator(['Real', 'Fake'])
results = evaluator.evaluate_predictions(y_true, y_pred, y_prob)

# Generate visualizations
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curve()
evaluator.plot_precision_recall_curve()
evaluator.plot_class_metrics()

# Generate report
report = evaluator.generate_report('evaluation_report.txt')
```

### 3. Inference

#### Single Prediction
```bash
python inference.py --model_path models/optimized_bangla_fake_news.pth \
                   --single_text "আপনার খবর এখানে" \
                   --single_image images/sample.jpg
```

#### Batch Prediction
```bash
python inference.py --model_path models/optimized_bangla_fake_news.pth \
                   --data_path Test.csv \
                   --output_path predictions.csv \
                   --batch_size 32
```

#### Python API
```python
from inference import RealTimeInference

# Initialize
inference = RealTimeInference('models/optimized_bangla_fake_news.pth')

# Predict
result = inference.predict("খবর টেক্সট", "image_path.jpg")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## Performance Optimization

### Training Optimization

1. **Mixed Precision Training** (if supported):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(input_ids, attention_mask, images)
    loss = criterion(outputs, labels)
```

2. **Data Loading Optimization**:
```python
# Increase num_workers for faster data loading
train_loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)
```

3. **Model Parallelism** (for large models):
```python
model = nn.DataParallel(model)  # Multi-GPU
```

### Inference Optimization

1. **Model Quantization**:
```python
import torch.quantization as quantization

quantized_model = quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

2. **Batch Processing**:
```python
# Use larger batch sizes for inference
batch_inference.predict_batch(data, batch_size=64)
```

## Research Features

### 1. Ablation Studies

Test different components:
```python
# Text-only model
text_only_model = BanglaBARTEncoder()

# Image-only model  
image_only_model = EfficientResNet50()

# Full multimodal model
full_model = OptimizedMultimodalModel()
```

### 2. Feature Analysis

```python
# Extract features for analysis
with torch.no_grad():
    text_features = model.text_encoder(input_ids, attention_mask)
    image_features = model.image_encoder(images)
    fused_features = model.cross_attention(text_features, image_features)
```

### 3. Hyperparameter Tuning

Use tools like Optuna or Ray Tune:
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    # Train model with these hyperparameters
    # Return validation accuracy
    return validation_accuracy
```

## Monitoring and Logging

### TensorBoard
```bash
tensorboard --logdir runs
```

### Weights & Biases (Optional)
```python
import wandb

wandb.init(project="bangla-fake-news")
wandb.log({"accuracy": accuracy, "loss": loss})
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**:
   - Increase `num_workers` in DataLoader
   - Use `pin_memory=True`
   - Enable mixed precision

3. **Poor Performance**:
   - Check data preprocessing
   - Adjust learning rates
   - Increase model capacity

### Debug Mode

```python
# Enable debug mode for detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## File Structure

```
BanglaFakeNewsProject/
├── optimized_model.py      # Main model architecture
├── train_optimized.py      # Training pipeline
├── data_loader.py          # Data loading and preprocessing
├── app.py                  # Gradio demo application
├── inference.py           # Inference scripts
├── evaluation.py          # Evaluation tools
├── requirements.txt       # Dependencies
├── config.yaml            # Configuration file
├── README.md             # This file
├── models/               # Saved models
├── checkpoints/          # Training checkpoints
├── runs/                 # TensorBoard logs
└── images/               # Image data
```

## Citation

If you use this work in your research, please cite:

```bibtex
@thesis{bangla_fake_news_2024,
  title={Optimized Multimodal Bangla Fake News Detection using BART and ResNet50},
  author={Your Name},
  year={2024},
  institution={Your University}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Bangla BART model from CSE BUET NLP
- ResNet architecture from torchvision
- Transformers library from Hugging Face
- Research community for datasets and benchmarks
