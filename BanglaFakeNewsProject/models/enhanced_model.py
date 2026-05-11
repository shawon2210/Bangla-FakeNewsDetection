"""
Enhanced Multimodal Model with Advanced Fusion and Attention Mechanisms
Improved architecture for higher accuracy in Bangla fake news detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Import specific submodules to speed up Transformers import on CPU-only setups
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from torchvision import models
import math

class MultiHeadCrossAttention(nn.Module):
    """Enhanced multi-head cross-attention with residual connections"""
    
    def __init__(self, text_dim, image_dim, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projections
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, text_features, image_features):
        # Project to common space
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features.unsqueeze(1))
        
        # Cross-attention: text attends to image
        attended, attention_weights = self.attention(text_proj, image_proj, image_proj)
        
        # Residual connection and layer norm
        attended = self.layer_norm1(attended + text_proj)
        
        # Feed-forward with residual
        ffn_out = self.ffn(attended)
        output = self.layer_norm2(ffn_out + attended)
        
        # Global pooling
        pooled = torch.mean(output, dim=1)
        
        return pooled, attention_weights

class AdaptiveFusion(nn.Module):
    """Adaptive fusion mechanism that learns optimal combination weights"""
    
    def __init__(self, text_dim, image_dim, fusion_dim=512):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.fusion_dim = fusion_dim
        
        # Gating mechanism
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        self.image_gate = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.text_transform = nn.Linear(text_dim, fusion_dim)
        self.image_transform = nn.Linear(image_dim, fusion_dim)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, text_features, image_features):
        # Compute adaptive weights
        text_weight = self.text_gate(text_features)
        image_weight = self.image_gate(image_features)
        
        # Transform features
        text_transformed = self.text_transform(text_features) * text_weight
        image_transformed = self.image_transform(image_features) * image_weight
        
        # Concatenate and fuse
        combined = torch.cat([text_transformed, image_transformed], dim=1)
        fused = self.fusion_layers(combined)
        
        return fused

class EnhancedTextEncoder(nn.Module):
    """Enhanced text encoder with better feature extraction"""
    
    def __init__(self, model_name="csebuetnlp/banglabert", freeze_layers=4):
        super().__init__()
        
        # Load model with error handling
        try:
            self.config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except:
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        # Selective freezing for better adaptation
        if freeze_layers > 0:
            for i, layer in enumerate(self.model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Multi-scale feature extraction
        self.feature_layers = nn.ModuleList([
            nn.Linear(self.config.hidden_size, 512),
            nn.Linear(self.config.hidden_size, 512),
            nn.Linear(self.config.hidden_size, 512)
        ])
        
        self.attention_pooling = nn.MultiheadAttention(512, 8, batch_first=True)
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract features from multiple layers
        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1]
        second_last = hidden_states[-2]
        third_last = hidden_states[-3]
        
        # Transform features
        feat1 = self.feature_layers[0](last_hidden)
        feat2 = self.feature_layers[1](second_last)
        feat3 = self.feature_layers[2](third_last)
        
        # Combine features with attention
        combined = torch.stack([feat1, feat2, feat3], dim=2)  # [batch, seq, 3, 512]
        batch_size, seq_len, num_layers, hidden_dim = combined.shape
        combined = combined.view(batch_size, seq_len * num_layers, hidden_dim)
        
        # Self-attention pooling
        pooled, _ = self.attention_pooling(combined, combined, combined)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        # Global average pooling
        output = torch.mean(pooled, dim=1)
        
        return output

class EnhancedImageEncoder(nn.Module):
    """Enhanced image encoder with attention and multi-scale features"""
    
    def __init__(self, pretrained=True, feature_dim=512):
        super().__init__()
        
        # Load ResNet50 backbone
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Extract intermediate features
        self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])
        self.layer2 = nn.Sequential(*list(self.backbone.children())[5:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[6:7])
        self.layer4 = nn.Sequential(*list(self.backbone.children())[7:8])
        
        # Adaptive pooling for different scales
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Final projection
        self.feature_proj = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract multi-scale features
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply spatial attention
        attention_map = self.spatial_attention(x)
        x = x * attention_map
        
        # Feature fusion and pooling
        x = self.feature_fusion(x)
        x = x.view(x.size(0), -1)
        
        # Final projection
        features = self.feature_proj(x)
        
        return features

class EnhancedMultimodalModel(nn.Module):
    """Enhanced multimodal model with advanced fusion and attention mechanisms"""
    
    def __init__(self, 
                 text_model_name="csebuetnlp/banglabert",
                 num_classes=2,
                 text_feature_dim=512,
                 image_feature_dim=512,
                 fusion_dim=512,
                 dropout_rate=0.2):
        super().__init__()
        
        # Enhanced encoders
        self.text_encoder = EnhancedTextEncoder(text_model_name, freeze_layers=3)
        self.image_encoder = EnhancedImageEncoder(pretrained=True, feature_dim=image_feature_dim)
        
        # Cross-modal attention
        self.cross_attention = MultiHeadCrossAttention(
            text_feature_dim, image_feature_dim, fusion_dim, num_heads=8
        )
        
        # Adaptive fusion
        self.adaptive_fusion = AdaptiveFusion(
            text_feature_dim, image_feature_dim, fusion_dim
        )
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2 + text_feature_dim + image_feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize classifier weights with Xavier initialization"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask, images):
        # Extract features
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)
        
        # Cross-modal attention
        attended_features, attention_weights = self.cross_attention(
            text_features.unsqueeze(1), image_features
        )
        
        # Adaptive fusion
        fused_features = self.adaptive_fusion(text_features, image_features)
        
        # Combine all features
        combined_features = torch.cat([
            text_features,
            image_features,
            attended_features,
            fused_features
        ], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output

# Model factory function
def create_enhanced_model(config):
    """Create enhanced model from configuration"""
    return EnhancedMultimodalModel(
        text_model_name=config['model']['text_model_name'],
        num_classes=config['model']['num_classes'],
        text_feature_dim=config['model']['text_feature_dim'],
        image_feature_dim=config['model']['image_feature_dim'],
        fusion_dim=config['model']['fusion_dim'],
        dropout_rate=config['model']['dropout_rate']
    )

if __name__ == "__main__":
    # Test model creation
    model = EnhancedMultimodalModel()
    print("Enhanced model created successfully!")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    images = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask, images)
        print(f"Output shape: {output.shape}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")