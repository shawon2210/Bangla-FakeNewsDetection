"""
Enhanced Predictor with Ensemble Methods and Real-time Performance Monitoring
Improved prediction accuracy through multiple model ensemble
"""

import sys
from pathlib import Path

# Add parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
import numpy as np
import time
from transformers import AutoTokenizer
from enhanced_model import EnhancedMultimodalModel
from model_defs import OptimizedMultimodalModel, MultimodalFakeNewsDetector
import os
import json
from datetime import datetime

class EnhancedPredictor:
    """Enhanced predictor with ensemble methods and confidence calibration"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_dir = os.path.dirname(config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = self._select_device(self.config.get('hardware', {}).get('device', 'auto'))
        print(f"🔧 Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        self.image_transform = self._get_image_transform()
        
        # Load ensemble of models
        self.models = self._load_ensemble_models()
        
        # Performance tracking
        self.prediction_history = []
        self.performance_stats = {
            'total_predictions': 0,
            'avg_inference_time': 0.0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
    def _select_device(self, prefer: str):
        """Select optimal device"""
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
        """Load tokenizer with error handling"""
        try:
            return AutoTokenizer.from_pretrained(
                self.config['model']['text_model_name'], 
                local_files_only=True
            )
        except:
            return AutoTokenizer.from_pretrained(self.config['model']['text_model_name'])
    
    def _get_image_transform(self):
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_ensemble_models(self):
        """Load ensemble of different models for better accuracy"""
        models = {}
        
        # Model paths
        enhanced_model_path = os.path.join(self.config_dir, 'outputs', 'enhanced_best_model.pth')
        optimized_model_path = os.path.join(self.config_dir, self.config['logging']['model_save_path'])
        multimodal_v1_path = os.path.join(self.config_dir, self.config['logging']['multimodal_detector_v1_path'])
        
        # Load Enhanced Model (if available) - actually an OptimizedMultimodalModel copy
        if os.path.exists(enhanced_model_path):
            try:
                print("📦 Loading Enhanced Model...")
                # The enhanced_best_model.pth is a copy of the optimized model
                model = OptimizedMultimodalModel(
                    text_model_name=self.config['model']['text_model_name'],
                    num_classes=self.config['model']['num_classes']
                ).to(self.device)
                
                state_dict = torch.load(enhanced_model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Handle key mapping
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
                models['enhanced'] = model
                print("✅ Enhanced Model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load Enhanced Model: {e}")
        
        # Load Optimized Model
        if os.path.exists(optimized_model_path):
            try:
                print("📦 Loading Optimized Model...")
                model = OptimizedMultimodalModel(
                    text_model_name=self.config['model']['text_model_name'],
                    num_classes=self.config['model']['num_classes']
                ).to(self.device)
                
                state_dict = torch.load(optimized_model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Handle key mapping
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
                models['optimized'] = model
                print("✅ Optimized Model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load Optimized Model: {e}")
        
        # Load Multimodal V1 Model
        if os.path.exists(multimodal_v1_path):
            try:
                print("📦 Loading Multimodal V1 Model...")
                model = MultimodalFakeNewsDetector(
                    num_classes=self.config['model']['num_classes']
                ).to(self.device)
                
                state_dict = torch.load(multimodal_v1_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                models['multimodal_v1'] = model
                print("✅ Multimodal V1 Model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load Multimodal V1 Model: {e}")
        
        if not models:
            raise RuntimeError("No models could be loaded! Please check model paths.")
        
        print(f"🎯 Ensemble ready with {len(models)} models: {list(models.keys())}")
        return models
    
    def _resolve_image_path(self, image_path: str | None) -> str | None:
        """Resolve image path with multiple extension support"""
        if not image_path:
            return None
        if os.path.exists(image_path):
            return image_path
        
        # Try different extensions
        base, ext = os.path.splitext(image_path)
        for e in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.jfif']:
            candidate = base + e
            if os.path.exists(candidate):
                return candidate
        return None
    
    def _preprocess_inputs(self, text, image_path):
        """Preprocess text and image inputs"""
        # Preprocess text
        encodings = self.tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=self.config['data']['max_text_length'],
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Preprocess image
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
        
        return input_ids, attention_mask, image
    
    def _ensemble_predict(self, input_ids, attention_mask, image):
        """Perform ensemble prediction with confidence calibration"""
        all_probabilities = []
        model_predictions = {}
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                try:
                    outputs = model(input_ids, attention_mask, image)
                    probabilities = F.softmax(outputs, dim=1)
                    all_probabilities.append(probabilities)
                    
                    # Store individual model predictions
                    confidence, predicted_class = torch.max(probabilities, 1)
                    model_predictions[model_name] = {
                        'probabilities': probabilities.cpu().numpy().tolist()[0],
                        'prediction': predicted_class.item(),
                        'confidence': confidence.item()
                    }
                    
                except Exception as e:
                    print(f"⚠️ Error with model {model_name}: {e}")
                    continue
        
        if not all_probabilities:
            raise RuntimeError("All models failed to make predictions!")
        
        # Ensemble averaging with weighted combination
        weights = [1.0] * len(all_probabilities)  # Equal weights for now
        if len(all_probabilities) > 1:
            # Give more weight to enhanced model if available
            if 'enhanced' in self.models and len(all_probabilities) >= 2:
                weights[0] = 1.5  # Assuming enhanced model is first
        
        # Weighted average
        weighted_probs = torch.zeros_like(all_probabilities[0])
        total_weight = sum(weights)
        
        for prob, weight in zip(all_probabilities, weights):
            weighted_probs += prob * (weight / total_weight)
        
        # Final prediction
        confidence, predicted_class = torch.max(weighted_probs, 1)
        
        return {
            'ensemble_probabilities': weighted_probs.cpu().numpy().tolist()[0],
            'ensemble_prediction': predicted_class.item(),
            'ensemble_confidence': confidence.item(),
            'individual_models': model_predictions
        }
    
    def _calibrate_confidence(self, confidence, probabilities):
        """Calibrate confidence score for better reliability"""
        # Simple temperature scaling (can be improved with proper calibration)
        temperature = 1.2
        calibrated_probs = np.array(probabilities) ** (1/temperature)
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        
        # Entropy-based uncertainty
        entropy = -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-8))
        max_entropy = np.log(len(calibrated_probs))
        uncertainty = entropy / max_entropy
        
        # Adjusted confidence
        adjusted_confidence = confidence * (1 - uncertainty * 0.3)
        
        return adjusted_confidence, calibrated_probs.tolist()
    
    def detect_news_type(self, text):
        """Enhanced news type detection with more categories"""
        text_lower = str(text).lower()
        
        categories = {
            "Politics": ["রাজনীতি", "politics", "minister", "government", "election", "সংসদ", "নেতা", "দল"],
            "Business": ["অর্থনীতি", "business", "stock", "market", "company", "ব্যবসা", "টাকা", "বিনিয়োগ"],
            "Crime": ["ক্রাইম", "crime", "murder", "robbery", "assault", "হত্যা", "চুরি", "অপরাধ"],
            "Sports": ["খেলা", "sports", "match", "football", "cricket", "olympic", "game", "ফুটবল"],
            "Entertainment": ["বিনোদন", "entertainment", "movie", "film", "actor", "actress", "music", "সিনেমা"],
            "Health": ["স্বাস্থ্য", "health", "hospital", "doctor", "covid", "vaccine", "চিকিৎসা"],
            "Technology": ["প্রযুক্তি", "technology", "tech", "internet", "ai", "software", "কম্পিউটার"],
            "Education": ["শিক্ষা", "education", "school", "university", "student", "বিশ্ববিদ্যালয়"],
            "Religion": ["ধর্ম", "religion", "islam", "muslim", "মসজিদ", "নামাজ", "ইসলাম"]
        }
        
        for category, keywords in categories.items():
            if any(word in text_lower for word in keywords):
                return category
        
        return "General"
    
    def predict(self, text, image_path=None):
        """Enhanced prediction with ensemble and monitoring"""
        start_time = time.time()
        
        try:
            # Preprocess inputs
            input_ids, attention_mask, image = self._preprocess_inputs(text, image_path)
            
            # Ensemble prediction
            results = self._ensemble_predict(input_ids, attention_mask, image)
            
            # Calibrate confidence
            calibrated_confidence, calibrated_probs = self._calibrate_confidence(
                results['ensemble_confidence'], 
                results['ensemble_probabilities']
            )
            
            # Detect news type
            news_type = self.detect_news_type(text)
            
            # Inference time
            inference_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance_stats(calibrated_confidence, inference_time)
            
            # Prepare final result
            labels = ['Real', 'Fake']
            final_result = {
                "prediction": labels[results['ensemble_prediction']],
                "confidence": calibrated_confidence,
                "probabilities": calibrated_probs,
                "news_type": news_type,
                "inference_time": inference_time,
                "ensemble_details": results['individual_models'],
                "model_count": len(self.models)
            }
            
            # Log prediction
            self._log_prediction(text, image_path, final_result)
            
            return final_result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
                "news_type": "Unknown",
                "error": str(e)
            }
    
    def _update_performance_stats(self, confidence, inference_time):
        """Update performance statistics"""
        self.performance_stats['total_predictions'] += 1
        
        # Update average inference time
        total = self.performance_stats['total_predictions']
        current_avg = self.performance_stats['avg_inference_time']
        self.performance_stats['avg_inference_time'] = (current_avg * (total - 1) + inference_time) / total
        
        # Update confidence distribution
        if confidence >= 0.8:
            self.performance_stats['confidence_distribution']['high'] += 1
        elif confidence >= 0.6:
            self.performance_stats['confidence_distribution']['medium'] += 1
        else:
            self.performance_stats['confidence_distribution']['low'] += 1
    
    def _log_prediction(self, text, image_path, result):
        """Log prediction for analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(str(text)),
            'has_image': image_path is not None,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'news_type': result['news_type'],
            'inference_time': result['inference_time']
        }
        
        self.prediction_history.append(log_entry)
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def get_performance_report(self):
        """Generate performance report"""
        if not self.prediction_history:
            return "No predictions made yet."
        
        total = len(self.prediction_history)
        real_count = sum(1 for p in self.prediction_history if p['prediction'] == 'Real')
        fake_count = total - real_count
        
        avg_confidence = np.mean([p['confidence'] for p in self.prediction_history])
        avg_inference_time = self.performance_stats['avg_inference_time']
        
        conf_dist = self.performance_stats['confidence_distribution']
        
        report = f"""
📊 Enhanced Predictor Performance Report
========================================
Total Predictions: {self.performance_stats['total_predictions']}
Real News: {real_count} ({real_count/total*100:.1f}%)
Fake News: {fake_count} ({fake_count/total*100:.1f}%)

Average Confidence: {avg_confidence:.3f}
Average Inference Time: {avg_inference_time:.3f}s

Confidence Distribution:
- High (≥0.8): {conf_dist['high']} ({conf_dist['high']/total*100:.1f}%)
- Medium (0.6-0.8): {conf_dist['medium']} ({conf_dist['medium']/total*100:.1f}%)
- Low (<0.6): {conf_dist['low']} ({conf_dist['low']/total*100:.1f}%)

Models in Ensemble: {len(self.models)}
Active Models: {', '.join(self.models.keys())}
        """
        
        return report
    
    def save_performance_log(self, filepath="outputs/prediction_performance.json"):
        """Save performance log to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        log_data = {
            'performance_stats': self.performance_stats,
            'recent_predictions': self.prediction_history[-100:],  # Last 100 predictions
            'model_info': {
                'model_count': len(self.models),
                'model_names': list(self.models.keys()),
                'device': str(self.device)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📁 Performance log saved to {filepath}")

if __name__ == "__main__":
    # Test enhanced predictor
    predictor = EnhancedPredictor()
    
    # Test prediction
    test_text = "ভূমিকম্পে কেঁপে উঠলো ঢাকা। আজ সকালে রিখটার স্কেলে ৫.২ মাত্রার ভূমিকম্প অনুভূত হয়েছে।"
    result = predictor.predict(test_text)
    
    print("🧪 Test Prediction Result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"News Type: {result['news_type']}")
    print(f"Inference Time: {result['inference_time']:.3f}s")
    
    # Performance report
    print(predictor.get_performance_report())