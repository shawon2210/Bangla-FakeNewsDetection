"""
Efficient Training Pipeline for Bangla Fake News Detection
Optimized for real-world deployment with comprehensive evaluation
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model_defs import OptimizedMultimodalModel, ModelManager
from preprocess_data import MultimodalDataset

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict()

class MetricsTracker:
    """Track and compute comprehensive evaluation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.labels = []
        self.losses = []
    
    def update(self, predictions, labels, loss=None):
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        if loss is not None:
            self.losses.append(loss.item())
    
    def compute_metrics(self):
        """Compute comprehensive evaluation metrics"""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'avg_loss': np.mean(self.losses) if self.losses else 0.0
        }
        
        return metrics

class OptimizedTrainer:
    """Optimized trainer with comprehensive evaluation and monitoring"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # Initialize optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize tracking
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 7),
            min_delta=config.get('early_stopping_delta', 0.001)
        )
        
        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'runs'))
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for different components"""
        # Different learning rates for different parts
        text_params = []
        image_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'text_encoder' in name:
                text_params.append(param)
            elif 'image_encoder' in name:
                image_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': text_params, 'lr': self.config.get('text_lr', 1e-5)},
            {'params': image_params, 'lr': self.config.get('image_lr', 1e-4)},
            {'params': other_params, 'lr': self.config.get('other_lr', 1e-4)}
        ], weight_decay=self.config.get('weight_decay', 0.01))
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = self.config.get('total_steps', 1000)
        warmup_steps = self.config.get('warmup_steps', 100)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return scheduler
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        self.metrics_tracker.reset()
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            predictions = torch.argmax(outputs, dim=1)
            self.metrics_tracker.update(predictions, labels, loss)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return self.metrics_tracker.compute_metrics()
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                predictions = torch.argmax(outputs, dim=1)
                self.metrics_tracker.update(predictions, labels, loss)
        
        return self.metrics_tracker.compute_metrics()
    
    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['avg_loss'])
            self.history['val_loss'].append(val_metrics['avg_loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1_weighted'])
            self.history['val_f1'].append(val_metrics['f1_weighted'])
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train', train_metrics['avg_loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['avg_loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/Train', train_metrics['f1_weighted'], epoch)
            self.writer.add_scalar('F1/Validation', val_metrics['f1_weighted'], epoch)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['avg_loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Train F1: {train_metrics['f1_weighted']:.4f}")
            print(f"Val Loss: {val_metrics['avg_loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val F1: {val_metrics['f1_weighted']:.4f}")
            
            # Early stopping
            if self.early_stopping(val_metrics['avg_loss'], self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch, val_metrics)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Save final model
        self.save_final_model()
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        checkpoint_path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        model_path = self.config.get('model_save_path', 'final_model.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save(self.model.state_dict(), model_path)
        print(f"Final model saved: {model_path}")
        
        # Save training history
        history_path = model_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved: {history_path}")

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 plot
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning curves comparison
    axes[1, 1].plot(history['train_acc'], label='Train Accuracy', alpha=0.7)
    axes[1, 1].plot(history['val_acc'], label='Validation Accuracy', alpha=0.7)
    axes[1, 1].plot(history['train_f1'], label='Train F1', alpha=0.7)
    axes[1, 1].plot(history['val_f1'], label='Validation F1', alpha=0.7)
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved: {save_path}")

def main():
    """Main training function"""
    # Configuration
    config = {
        'text_model_name': 'csebuetnlp/banglabert',
        'num_classes': 2,
        'batch_size': 16,
        'epochs': 20,
        'text_lr': 1e-5,
        'image_lr': 1e-4,
        'other_lr': 1e-4,
        'weight_decay': 0.01,
        'early_stopping_patience': 7,
        'early_stopping_delta': 0.001,
        'save_every': 5,
        'checkpoint_dir': 'checkpoints',
        'model_save_path': 'models/optimized_bangla_fake_news.pth',
        'log_dir': 'runs'
    }
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data (assuming preprocess_data.py is available)
    from preprocess_data import train_loader, val_loader, test_loader
    
    # Initialize model
    model = OptimizedMultimodalModel(
        text_model_name=config['text_model_name'],
        num_classes=config['num_classes']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(0.1 * total_steps)
    config['total_steps'] = total_steps
    config['warmup_steps'] = warmup_steps
    
    # Initialize trainer
    trainer = OptimizedTrainer(model, device, config)
    
    # Train model
    history = trainer.train(train_loader, val_loader, config['epochs'])
    
    # Plot training history
    plot_training_history(history)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = trainer.validate_epoch(test_loader)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1_weighted']:.4f}")
    
    # Save test results
    with open('test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
