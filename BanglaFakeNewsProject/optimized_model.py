"""
Optimized Multimodal Bangla Fake News Detection Model
Combines Bangla BART and ResNet50 for efficient real-world deployment
Author: Research Thesis Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchvision import models
import math

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for text-image fusion"""
    
    def __init__(self, text_dim, image_dim, hidden_dim=256):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Projections
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Attention weights
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, text_features, image_features):
        # Project features to common space
        text_proj = self.text_proj(text_features)  # [batch, seq_len, hidden_dim]
        image_proj = self.image_proj(image_features.unsqueeze(1))  # [batch, 1, hidden_dim]
        
        # Cross-attention: text attends to image
        attended_text, _ = self.attention(text_proj, image_proj, image_proj)
        
        # Global average pooling of attended text
        attended_features = torch.mean(attended_text, dim=1)  # [batch, hidden_dim]
        
        # Output projection
        output = self.output_proj(attended_features)
        
        return output

class BanglaBARTEncoder(nn.Module):
    """Optimized Bangla BART encoder with efficient feature extraction"""
    
    def __init__(self, model_name="csebuetnlp/banglabert", freeze_layers=6):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers for efficiency
        if freeze_layers > 0:
            for i, layer in enumerate(self.model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Feature projection
        self.feature_proj = nn.Linear(self.config.hidden_size, 512)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use last hidden state
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Project features
        projected = self.feature_proj(last_hidden)
        projected = self.dropout(projected)
        
        return projected

class EfficientResNet50(nn.Module):
    """Optimized ResNet50 with efficient feature extraction"""
    
    def __init__(self, pretrained=True, feature_dim=512):
        super().__init__()
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Freeze early layers for efficiency (optional)
        self._freeze_layers()
        
    def _freeze_layers(self):
        """Freeze early ResNet layers for efficiency"""
        for i, child in enumerate(self.backbone.children()):
            if i < 4:  # Freeze first 4 layers
                for param in child.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.feature_proj(features)
        return features

class OptimizedMultimodalModel(nn.Module):
    """
    Optimized multimodal model for Bangla fake news detection
    Combines Bangla BART and ResNet50 with cross-modal attention
    """
    
    def __init__(self, 
                 text_model_name="csebuetnlp/banglabert",
                 num_classes=2,
                 text_feature_dim=512,
                 image_feature_dim=512,
                 fusion_dim=256,
                 dropout_rate=0.3):
        super().__init__()
        
        # Text encoder
        self.text_encoder = BanglaBARTEncoder(text_model_name, freeze_layers=6)
        
        # Image encoder
        self.image_encoder = EfficientResNet50(pretrained=True, feature_dim=image_feature_dim)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(text_feature_dim, image_feature_dim, fusion_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + text_feature_dim + image_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask, images):
        # Extract text features
        text_features = self.text_encoder(input_ids, attention_mask)  # [batch, seq_len, text_feature_dim]
        text_pooled = torch.mean(text_features, dim=1)  # [batch, text_feature_dim]
        
        # Extract image features
        image_features = self.image_encoder(images)  # [batch, image_feature_dim]
        
        # Cross-modal attention
        attended_features = self.cross_attention(text_features, image_features)  # [batch, fusion_dim]
        
        # Combine all features
        combined_features = torch.cat([
            text_pooled,
            image_features,
            attended_features
        ], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output

class ModelManager:
    """Utility class for model management and inference"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path, model_name="csebuetnlp/banglabert", num_classes=2):
        """Load trained model"""
        self.model = OptimizedMultimodalModel(model_name, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def save_model(self, model, save_path):
        """Save trained model"""
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def predict(self, text, image_path, image_transform):
        """Make prediction on single sample"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Load and transform image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_transform(image).unsqueeze(0)
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
        
        return {
            'prediction': prediction.item(),
            'probabilities': probabilities.cpu().numpy()[0],
            'confidence': torch.max(probabilities).item()
        }

# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    model = OptimizedMultimodalModel()
    print("Model created successfully!")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    text_feature_dim = 512
    
    # Create dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids, attention_mask, images)
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
