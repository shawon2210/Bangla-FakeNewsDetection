"""
Complete Training Script for Bangla Fake News Detection
Trains on both text (headlines) and images simultaneously
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing preprocessing
from preprocess_data import MultimodalDataset, train_df, val_df, test_df, tokenizer

# ===========================
# Configuration
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL_NAME = "csebuetnlp/banglabert"
IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_TEXT_LENGTH = 128
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5

print(f"Using device: {DEVICE}")
print(f"Number of classes: {len(train_df['label'].unique())}")

# Import model architectures
from model_defs import MultimodalFakeNewsDetector

# ===========================
# Configuration
# ===========================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    return avg_loss, accuracy, precision, recall, f1

def train_model():
    """Main training function"""
    print("🚀 Starting multimodal training...")
    
    # Create datasets and dataloaders
    train_dataset = MultimodalDataset(train_df)
    val_dataset = MultimodalDataset(val_df)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = MultimodalFakeNewsDetector(num_classes=2).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_multimodal_model.pth')
            print(f"💾 New best model saved! Validation Accuracy: {val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score plot
    axes[1, 0].plot(history['val_f1'], label='Validation F1', marker='o', color='green')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined metrics
    axes[1, 1].plot(history['val_acc'], label='Accuracy', marker='o')
    axes[1, 1].plot(history['val_precision'], label='Precision', marker='s')
    axes[1, 1].plot(history['val_recall'], label='Recall', marker='^')
    axes[1, 1].plot(history['val_f1'], label='F1', marker='d')
    axes[1, 1].set_title('All Validation Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Training history plot saved as 'training_history.png'")

def test_model(model_path='best_multimodal_model.pth'):
    """Test the trained model"""
    print("\n🧪 Testing model on test set...")
    
    # Load best model
    model = MultimodalFakeNewsDetector(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Create test dataloader
    test_dataset = MultimodalDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_precision, test_recall, test_f1 = validate_epoch(
        model, test_loader, criterion, DEVICE
    )
    
    print(f"\n📊 Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    return test_acc

# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    print("🎯 Bangla Fake News Detection - Multimodal Training")
    print("=" * 60)
    
    # Check data
    print(f"📊 Dataset Info:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Classes: {sorted(train_df['label'].unique())}")
    
    # Train model
    trained_model, history = train_model()
    
    # Test model
    test_accuracy = test_model()
    
    print(f"\n🎉 Training completed successfully!")
    print(f"🏆 Final test accuracy: {test_accuracy:.4f}")
    print(f"💾 Best model saved as 'best_multimodal_model.pth'")
