"""
Unified Predictor - Seamless Integration of Enhanced Improvements with Existing Strategy
Maintains your proven working approach while adding enhanced capabilities
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
import numpy as np
import os
from transformers import AutoTokenizer

from model_defs import OptimizedMultimodalModel, MultimodalFakeNewsDetector

class UnifiedPredictor:
    """Unified predictor that combines existing proven strategy with enhancements"""
    
    def __init__(self, config_path="config.yaml"):
        # Handle different possible config locations
        if not os.path.exists(config_path):
            possible_paths = [
                "config.yaml",
                "../config.yaml", 
                os.path.join(os.path.dirname(__file__), "config.yaml"),
                os.path.join(os.path.dirname(__file__), "..", "config.yaml")
            ]
            
            config_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                raise FileNotFoundError("config.yaml not found in any expected location")
        
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self._select_device(self.config.get('hardware', {}).get('device', 'auto'))
        self.tokenizer = self._load_tokenizer()
        self.image_transform = self._get_image_transform()
        
        # Load all available models using existing strategy
        self.models = self._load_all_models()
        print(f"✅ Loaded {len(self.models)} models for ensemble prediction")
    
    def _select_device(self, prefer: str):
        """Existing device selection strategy"""
        prefer = (prefer or 'auto').lower()
        if prefer == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            d = torch.device(prefer)
            if d.type == 'cuda' and not torch.cuda.is_available():
                return torch.device('cpu')
            return d
        except Exception:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_tokenizer(self):
        """Existing tokenizer loading strategy"""
        try:
            return AutoTokenizer.from_pretrained(
                self.config['model']['text_model_name'], 
                local_files_only=True
            )
        except:
            return AutoTokenizer.from_pretrained(self.config['model']['text_model_name'])
    
    def _get_image_transform(self):
        """Existing image transform strategy"""
        return transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_all_models(self):
        """Load all available models using existing + enhanced strategies"""
        models = {}
        
        # 1. Load Enhanced Model (if available - currently a copy of optimized model)
        enhanced_path = os.path.join(self.config_dir, 'outputs', 'enhanced_best_model.pth')
        if os.path.exists(enhanced_path):
            try:
                # For now, enhanced model is a copy of OptimizedMultimodalModel
                model = OptimizedMultimodalModel(
                    text_model_name=self.config['model']['text_model_name'],
                    num_classes=self.config['model']['num_classes']
                ).to(self.device)
                
                state_dict = torch.load(enhanced_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Existing key mapping strategy
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('bert.'):
                        new_key = key.replace('bert.', 'text_encoder.model.')
                    elif key.startswith('resnet.'):
                        new_key = key.replace('resnet.', 'image_encoder.backbone.')
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                
                model.load_state_dict(new_state_dict, strict=False)
                model.eval()
                models['enhanced'] = {'model': model, 'weight': 2.0}  # Higher weight for enhanced
                print("✅ enhanced model loaded")
            except Exception as e:
                print(f"⚠️ enhanced model failed: {e}")
        
        # 2. Load Optimized Models (existing strategy)
        # Primary: Best trained model from train_multimodal.py or train_optimized.py
        # Secondary: Latest checkpoint for ensemble diversity
        model_paths = [
            ('primary', os.path.join(self.config_dir, self.config['logging']['model_save_path'])),
            ('secondary', os.path.join(self.config_dir, 'outputs', 'checkpoints', 'checkpoint_epoch_5.pth'))
        ]
        
        for name, path in model_paths:
            if os.path.exists(path):
                try:
                    model = OptimizedMultimodalModel(
                        text_model_name=self.config['model']['text_model_name'],
                        num_classes=self.config['model']['num_classes']
                    ).to(self.device)
                    
                    state_dict = torch.load(path, map_location=self.device, weights_only=False)
                    if 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    
                    # Existing key mapping strategy
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('bert.'):
                            new_key = key.replace('bert.', 'text_encoder.model.')
                        elif key.startswith('resnet.'):
                            new_key = key.replace('resnet.', 'image_encoder.backbone.')
                        else:
                            new_key = key
                        new_state_dict[new_key] = value
                    
                    model.load_state_dict(new_state_dict, strict=False)
                    model.eval()
                    models[name] = {'model': model, 'weight': 1.0}
                    print(f"✅ {name} model loaded")
                except Exception as e:
                    print(f"⚠️ {name} model failed: {e}")
        
        # 3. Load MultimodalFakeNewsDetector (BERT+LSTM+Attention architecture)
        v1_path = os.path.join(self.config_dir, self.config['logging']['multimodal_detector_v1_path'])
        if os.path.exists(v1_path):
            try:
                model = MultimodalFakeNewsDetector(
                    num_classes=self.config['model']['num_classes']
                ).to(self.device)
                
                state_dict = torch.load(v1_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                models['multimodal_v1'] = {'model': model, 'weight': 1.0}
                print("✅ multimodal_v1 model loaded")
            except Exception as e:
                print(f"⚠️ multimodal_v1 model failed: {e}")
        
        return models
    
    def _resolve_image_path(self, image_path: str | None) -> str | None:
        """Existing image path resolution strategy"""
        if not image_path:
            return None
        if os.path.exists(image_path):
            return image_path
        
        base, ext = os.path.splitext(image_path)
        for e in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.jfif']:
            candidate = base + e
            if os.path.exists(candidate):
                return candidate
        return None
    
    def detect_news_type(self, text):
        """Enhanced news type detection (existing + improved)"""
        text_lower = str(text).lower()
        
        # Existing categories + enhanced detection
        categories = {
            "Politics": ["রাজনীতি", "politics", "minister", "government", "election", "সংসদ", "নেতা", "দল", "ভোট"],
            "Business": ["অর্থনীতি", "business", "stock", "market", "company", "ব্যবসা", "টাকা", "বিনিয়োগ", "ব্যাংক"],
            "Crime": ["ক্রাইম", "crime", "murder", "robbery", "assault", "হত্যা", "চুরি", "অপরাধ", "পুলিশ"],
            "Sports": ["খেলা", "sports", "match", "football", "cricket", "olympic", "game", "ফুটবল", "ক্রিকেট"],
            "Entertainment": ["বিনোদন", "entertainment", "movie", "film", "actor", "actress", "music", "সিনেমা"],
            "Health": ["স্বাস্থ্য", "health", "hospital", "doctor", "covid", "vaccine", "চিকিৎসা", "রোগ"],
            "Technology": ["প্রযুক্তি", "technology", "tech", "internet", "ai", "software", "কম্পিউটার"],
            "Education": ["শিক্ষা", "education", "school", "university", "student", "বিশ্ববিদ্যালয়", "পরীক্ষা"]
        }
        
        for category, keywords in categories.items():
            if any(word in text_lower for word in keywords):
                return category
        
        return "General"
    
    def predict(self, text, image_path=None):
        """Unified prediction using existing strategy + enhancements"""
        
        # Existing preprocessing strategy
        encodings = self.tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=self.config['data']['max_text_length'],
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Existing image preprocessing
        img_tensor = None
        resolved_path = self._resolve_image_path(image_path)
        if resolved_path:
            try:
                img = Image.open(resolved_path).convert("RGB")
                img_tensor = self.image_transform(img).unsqueeze(0)
            except Exception as e:
                print(f"[WARNING] Failed to load image '{resolved_path}': {e}")
        
        if img_tensor is None:
            img_tensor = torch.zeros(1, 3, self.config['data']['image_size'], self.config['data']['image_size'])
        
        image = img_tensor.to(self.device)
        
        # Enhanced ensemble prediction with existing averaging strategy
        all_probabilities = []
        weights = []
        model_results = {}
        
        with torch.no_grad():
            for name, model_info in self.models.items():
                try:
                    model = model_info['model']
                    weight = model_info['weight']
                    
                    outputs = model(input_ids, attention_mask, image)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    all_probabilities.append(probabilities)
                    weights.append(weight)
                    
                    # Store individual results (existing strategy)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    model_results[name] = {
                        'probabilities': probabilities.cpu().numpy().tolist()[0],
                        'prediction': predicted_class.item(),
                        'confidence': confidence.item()
                    }
                    
                    print(f"{name} probabilities: {probabilities.cpu().numpy().tolist()[0]}")
                    
                except Exception as e:
                    print(f"⚠️ Error with {name}: {e}")
                    continue
        
        if not all_probabilities:
            raise RuntimeError("All models failed!")
        
        # Enhanced weighted averaging (improved from existing simple average)
        total_weight = sum(weights)
        weighted_probs = torch.zeros_like(all_probabilities[0])
        
        for prob, weight in zip(all_probabilities, weights):
            weighted_probs += prob * (weight / total_weight)
        
        # Existing confidence calculation strategy
        confidence, predicted_class = torch.max(weighted_probs, 1)
        
        # Enhanced confidence calibration (optional improvement)
        calibrated_confidence = self._calibrate_confidence(confidence.item(), weighted_probs.cpu().numpy()[0])
        
        # Existing news type detection + enhancement
        news_type = self.detect_news_type(text)
        
        # Enhanced result format (backward compatible)
        labels = ['Real', 'Fake']
        result = {
            "prediction": labels[predicted_class.item()],
            "confidence": calibrated_confidence,
            "probabilities": weighted_probs.cpu().numpy().tolist()[0],
            "news_type": news_type,
            "model_count": len(self.models),
            "individual_models": model_results  # Enhanced detail
        }
        
        print(f"Final averaged probabilities: {result['probabilities']}")
        print(f"Final prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return result
    
    def _calibrate_confidence(self, confidence, probabilities):
        """Enhanced confidence calibration (optional improvement)"""
        # Simple temperature scaling + entropy adjustment
        temperature = 1.1
        calibrated_probs = np.array(probabilities) ** (1/temperature)
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        
        # Entropy-based uncertainty (enhancement)
        entropy = -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-8))
        max_entropy = np.log(len(calibrated_probs))
        uncertainty = entropy / max_entropy
        
        # Adjust confidence based on uncertainty
        adjusted_confidence = confidence * (1 - uncertainty * 0.2)
        
        return max(adjusted_confidence, 0.1)  # Minimum confidence threshold

# Backward compatibility wrapper
class Predictor(UnifiedPredictor):
    """Backward compatible wrapper maintaining exact same interface"""
    pass

if __name__ == "__main__":
    # Test unified predictor
    predictor = UnifiedPredictor()
    
    test_text = "ভূমিকম্পে কেঁপে উঠলো ঢাকা। আজ সকালে রিখটার স্কেলে ৫.২ মাত্রার ভূমিকম্প অনুভূত হয়েছে।"
    result = predictor.predict(test_text)
    
    print("\n🧪 Unified Prediction Result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"News Type: {result['news_type']}")
    print(f"Models Used: {result['model_count']}")