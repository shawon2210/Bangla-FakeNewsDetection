import torch
from torchvision import transforms
from PIL import Image
import yaml
from transformers import AutoTokenizer
from model_defs import OptimizedMultimodalModel, MultimodalFakeNewsDetector

import os

class Predictor:
    def detect_news_type(self, text):
        text_lower = str(text).lower()
        # Simple keyword-based rules (expand as needed)
        if any(word in text_lower for word in ["রাজনীতি", "politics", "minister", "government", "election", "সংসদ"]):
            return "Politics"
        elif any(word in text_lower for word in ["অর্থনীতি", "business", "stock", "market", "company", "ব্যবসা"]):
            return "Business"
        elif any(word in text_lower for word in ["ক্রাইম", "crime", "murder", "robbery", "assault", "হত্যা", "চুরি"]):
            return "Crime"
        elif any(word in text_lower for word in ["খেলা", "sports", "match", "football", "cricket", "olympic", "game"]):
            return "Sports"
        elif any(word in text_lower for word in ["বিনোদন", "entertainment", "movie", "film", "actor", "actress", "music"]):
            return "Entertainment"
        elif any(word in text_lower for word in ["স্বাস্থ্য", "health", "hospital", "doctor", "covid", "vaccine"]):
            return "Health"
        elif any(word in text_lower for word in ["প্রযুক্তি", "technology", "tech", "internet", "ai", "software"]):
            return "Technology"
        else:
            return "General"
            
    def __init__(self, config_path="config.yaml"):
        self.config_dir = os.path.dirname(config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Robust device selection: honor 'auto' and fall back safely
        self.device = self._select_device(self.config.get('hardware', {}).get('device', 'auto'))

        # Define model paths (use safe defaults when config keys are missing)
        logging_cfg = self.config.get('logging', {})
        primary_model_path = os.path.join(self.config_dir, logging_cfg.get('model_save_path', 'best_multimodal_model.pth'))
        secondary_model_path = os.path.join(self.config_dir, 'outputs', 'checkpoints', 'checkpoint_epoch_5.pth')
        multimodal_detector_v1_path = os.path.join(self.config_dir, logging_cfg.get('multimodal_detector_v1_path', 'best_multimodal_model.pth'))
        enhanced_model_path = os.path.join(self.config_dir, 'outputs', 'enhanced_best_model.pth')

        # Load models for ensembling
        self.models = []
        self.model_names = []

        # Model 1: Primary optimized model
        if os.path.exists(primary_model_path):
            print("Loading primary model...")
            self.models.append(self._load_optimized_model(primary_model_path))
            self.model_names.append("Primary Optimized")

        # Model 2: Checkpoint model
        if os.path.exists(secondary_model_path):
            print("Loading checkpoint model...")
            self.models.append(self._load_optimized_model(secondary_model_path))
            self.model_names.append("Checkpoint")

        # Model 3: MultimodalFakeNewsDetector V1
        if os.path.exists(multimodal_detector_v1_path):
            print("Loading MultimodalFakeNewsDetector V1...")
            self.models.append(self._load_multimodal_detector_v1_model(multimodal_detector_v1_path))
            self.model_names.append("Multimodal V1")

        # Model 4: Enhanced model (if available)
        if os.path.exists(enhanced_model_path):
            print("Loading enhanced model...")
            self.models.append(self._load_optimized_model(enhanced_model_path))
            self.model_names.append("Enhanced")

        if not self.models:
            raise RuntimeError("No models could be loaded! Please check model paths.")

        print(f"✅ Loaded {len(self.models)} models: {', '.join(self.model_names)}")

        self.tokenizer = self._load_tokenizer()
        self.image_transform = self._get_image_transform()

    def _select_device(self, prefer: str):
        """Select a torch.device based on preference and availability.
        prefer can be 'auto', 'cuda', 'cpu'."""
        prefer = (prefer or 'auto').lower()
        if prefer == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            d = torch.device(prefer)
            # if 'cuda' requested but unavailable, fall back
            if d.type == 'cuda' and not torch.cuda.is_available():
                return torch.device('cpu')
            return d
        except Exception:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_optimized_model(self, model_path):
        model = OptimizedMultimodalModel(
            text_model_name=self.config['model']['text_model_name'],
            num_classes=self.config['model']['num_classes']
        ).to(self.device)
        
        print(f"Loading OptimizedMultimodalModel from: {model_path}")
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # If it's a checkpoint, extract the model's state dict
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Create a new state dictionary with corrected keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_key = key.replace('bert.', 'text_encoder.model.')
            elif key.startswith('resnet.'):
                new_key = key.replace('resnet.', 'image_encoder.backbone.')
            else:
                new_key = key
            new_state_dict[new_key] = value
            
        # Load the new state dictionary
        model.load_state_dict(new_state_dict, strict=False)
        
        model.eval()
        return model

    def _load_multimodal_detector_v1_model(self, model_path):
        model = MultimodalFakeNewsDetector(
            num_classes=self.config['model']['num_classes']
        ).to(self.device)

        print(f"Loading MultimodalFakeNewsDetector from: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config['model']['text_model_name'])

    def _get_image_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _resolve_image_path(self, image_path: str | None) -> str | None:
        """Return an existing image path by probing common extensions if needed.
        If the given path exists, return it. If it doesn't and it looks like
        an images/* path with a base name, try alternate extensions.
        Returns None if nothing is found."""
        if not image_path:
            return None
        if os.path.exists(image_path):
            return image_path
        # Try swapping/adding common extensions
        base, ext = os.path.splitext(image_path)
        candidates = []
        # If ext missing or wrong, try known ones
        for e in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.jfif']:
            candidates.append(base + e)
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def predict(self, text, image_path):
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

        # Preprocess image (robust to missing/non-existent paths)
        img_tensor = None
        resolved_path = self._resolve_image_path(image_path)
        if resolved_path:
            try:
                img = Image.open(resolved_path).convert("RGB")
                img_tensor = self.image_transform(img).unsqueeze(0)
            except Exception as e:
                print(f"[WARNING] Failed to load image '{resolved_path}': {e}. Using blank image.")
        if img_tensor is None:
            img_tensor = torch.zeros(1, 3, self.config['data']['image_size'], self.config['data']['image_size'])
        image = img_tensor.to(self.device)

        # Predict with all loaded models
        all_probabilities = []
        with torch.no_grad():
            for idx, model in enumerate(self.models):
                outputs = model(input_ids, attention_mask, image)
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.append(probabilities)
                print(f"Model {idx+1} ({self.model_names[idx]}) probabilities: {probabilities.cpu().numpy().tolist()[0]}")
            
            # Ensemble by averaging probabilities from all models
            avg_probabilities = torch.stack(all_probabilities).mean(dim=0)
            print(f"Averaged probabilities (from {len(self.models)} models): {avg_probabilities.cpu().numpy().tolist()[0]}")
            
            confidence, predicted_class = torch.max(avg_probabilities, 1)

        labels = ['Real', 'Fake']
        news_type = self.detect_news_type(text)
        return {
            "prediction": labels[predicted_class.item()],
            "confidence": confidence.item(),
            "probabilities": avg_probabilities.cpu().numpy().tolist()[0],
            "news_type": news_type,
            "model_count": len(self.models)
        }
