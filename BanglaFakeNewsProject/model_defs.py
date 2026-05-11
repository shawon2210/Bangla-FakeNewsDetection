"""
Model architecture definitions for Bangla Fake News Detection.

This module contains ONLY model classes — no training logic, no data loading.
Safe to import from anywhere (including predictor.py) without triggering
heavy dependencies.

Exports:
  - AttentionLayer: attention pooling for LSTM outputs
  - MultimodalFakeNewsDetector: v1 model (BERT + BiLSTM + Attention + ResNet50)
  - OptimizedMultimodalModel: v2 model (BanglaBART + ResNet50 + cross-attention)
  - CrossModalAttention: cross-modal attention mechanism
  - BanglaBARTEncoder: Bangla BART text encoder
  - EfficientResNet50: ResNet50 image encoder
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torchvision import models


class AttentionLayer(nn.Module):
    """Attention mechanism for text features."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        return torch.sum(attention_weights * lstm_output, dim=1)


class MultimodalFakeNewsDetector(nn.Module):
    """Multimodal model v1: BERT + BiLSTM + Attention + ResNet50."""

    TEXT_MODEL_NAME = "csebuetnlp/banglabert"

    def __init__(self, num_classes=2):
        super().__init__()

        self.bert = AutoModel.from_pretrained(self.TEXT_MODEL_NAME)
        for param in self.bert.encoder.layer[:6].parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=768, hidden_size=256, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.text_attention = AttentionLayer(512)

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in list(resnet.children())[:4]:
            for p in param.parameters():
                p.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Linear(2048, 512)

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_outputs.last_hidden_state)
        text_features = self.text_attention(lstm_output)

        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_fc(image_features)

        combined = torch.cat([text_features, image_features], dim=1)
        return self.fusion(combined)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for text-image fusion."""

    def __init__(self, text_dim, image_dim, hidden_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_features, image_features):
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features.unsqueeze(1))
        attended_text, _ = self.attention(text_proj, image_proj, image_proj)
        return self.output_proj(torch.mean(attended_text, dim=1))


class BanglaBARTEncoder(nn.Module):
    """Bangla BART encoder with frozen early layers."""

    def __init__(self, model_name="csebuetnlp/banglabert", freeze_layers=6):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if freeze_layers > 0:
            for i, layer in enumerate(self.model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        self.feature_proj = nn.Linear(self.config.hidden_size, 512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )
        return self.dropout(self.feature_proj(outputs.last_hidden_state))


class EfficientResNet50(nn.Module):
    """Optimized ResNet50 image encoder."""

    def __init__(self, pretrained=True, feature_dim=512):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.feature_proj = nn.Sequential(
            nn.Linear(2048, feature_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        self._freeze_layers()

    def _freeze_layers(self):
        for i, child in enumerate(self.backbone.children()):
            if i < 4:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        return self.feature_proj(features)


class OptimizedMultimodalModel(nn.Module):
    """Optimized multimodal model v2: BanglaBART + ResNet50 + cross-modal attention."""

    def __init__(self, text_model_name="csebuetnlp/banglabert", num_classes=2,
                 text_feature_dim=512, image_feature_dim=512, fusion_dim=256,
                 dropout_rate=0.3):
        super().__init__()
        self.text_encoder = BanglaBARTEncoder(text_model_name, freeze_layers=6)
        self.image_encoder = EfficientResNet50(pretrained=True, feature_dim=image_feature_dim)
        self.cross_attention = CrossModalAttention(text_feature_dim, image_feature_dim, fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + text_feature_dim + image_feature_dim, 512),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_encoder(input_ids, attention_mask)
        text_pooled = torch.mean(text_features, dim=1)
        image_features = self.image_encoder(images)
        attended = self.cross_attention(text_features, image_features)
        combined = torch.cat([text_pooled, image_features, attended], dim=1)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------
class ModelManager:
    """Utility for model loading, saving, and inference."""

    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path, model_name="csebuetnlp/banglabert", num_classes=2):
        from transformers import AutoTokenizer
        self.model = OptimizedMultimodalModel(model_name, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def save_model(self, model, save_path):
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
