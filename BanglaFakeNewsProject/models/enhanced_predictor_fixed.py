"""
Enhanced Predictor with Ensemble Methods and Real-time Performance Monitoring
Improved prediction accuracy through multiple model ensemble
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
import numpy as np
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to sys.path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# Also add models directory
MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from transformers import AutoTokenizer
from enhanced_model import EnhancedMultimodalModel
from model_defs import OptimizedMultimodalModel, MultimodalFakeNewsDetector
