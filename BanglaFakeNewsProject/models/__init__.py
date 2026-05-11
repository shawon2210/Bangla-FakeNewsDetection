"""
Model architectures for Bangla Fake News Detection.

Exports:
  - OptimizedMultimodalModel: primary model (BanglaBART + ResNet50 + cross-attention)
  - MultimodalFakeNewsDetector: v1 model (BERT + BiLSTM + Attention + ResNet50)
  - EnhancedMultimodalModel: advanced model (multi-scale features, adaptive fusion)
  - ModelManager: utility for loading/saving/running models
"""

from .optimized_model import OptimizedMultimodalModel, ModelManager
from .train_multimodal import MultimodalFakeNewsDetector
from .enhanced_model import EnhancedMultimodalModel

__all__ = [
    "OptimizedMultimodalModel",
    "MultimodalFakeNewsDetector",
    "EnhancedMultimodalModel",
    "ModelManager",
]
