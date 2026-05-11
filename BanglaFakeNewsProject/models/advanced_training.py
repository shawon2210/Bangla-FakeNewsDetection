"""
Advanced Training Pipeline with Real-time Monitoring and Improved Strategies
Enhanced training for better accuracy and efficiency
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import time
import os
import sys
import json
from datetime import datetime
from pathlib import Path
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to sys.path to import modules from project root
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Also add models directory for local imports
MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from enhanced_model import EnhancedMultimodalModel
from data_loader import MultimodalDataset
from preprocess_data import prepare_datasets_from_csv
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

class AdvancedTrainer:
    """Advanced trainer with real-time monitoring and optimization strategies"""
    
    def __init__(self, config_path=None):
        # Resolve config path relative to project root
        if config_path is None:
            config_path = str(BASE_DIR / "config.yaml")
        elif not os.path.isabs(config_path):
            config_path = str(BASE_DIR / config_path)
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f"runs/enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Training metrics
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
    def setup_data(self, csv_path=None):
        """Setup data loaders with enhanced preprocessing"""
        print("📊 Setting up data loaders...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['text_model_name'])

            
        # Image preprocessing
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Prepare datasets - use existing CSV files
        import pandas as pd
        
        if csv_path:
            train_df, val_df, test_df = prepare_datasets_from_csv(csv_path)
        else:
            # Load from existing split CSV files
            print("📁 Loading from split CSV files...")
            train_df = pd.read_csv(self.config['data']['train_csv'])
            val_df = pd.read_csv(self.config['data']['val_csv'])
            test_df = pd.read_csv(self.config['data']['test_csv'])
            print(f"✅ Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
        
        # Create datasets with correct parameters
        train_dataset = MultimodalDataset(
            train_df, self.tokenizer, self.image_transform, self.config
        )
        val_dataset = MultimodalDataset(
            val_df, self.tokenizer, self.image_transform, self.config
        )
        
        # Create data loaders
        num_workers = 0 if os.name == 'nt' else self.config['hardware']['num_workers']
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config['hardware']['pin_memory'] and torch.cuda.is_available()
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config['hardware']['pin_memory'] and torch.cuda.is_available()
        )
        
        print(f"✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        return train_df, val_df, test_df
    
    def setup_model(self):
        """Setup enhanced model with optimized configuration"""
        print("🏗️ Setting up enhanced model...")
        
        self.model = EnhancedMultimodalModel(
            text_model_name=self.config['model']['text_model_name'],
            num_classes=self.config['model']['num_classes'],
            text_feature_dim=self.config['model']['text_feature_dim'],
            image_feature_dim=self.config['model']['image_feature_dim'],
            fusion_dim=self.config['model']['fusion_dim'],
            dropout_rate=self.config['model']['dropout_rate']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return self.model
    
    def setup_optimizer_scheduler(self):
        """Setup advanced optimizer and learning rate scheduler"""
        print("⚙️ Setting up optimizer and scheduler...")
        
        # Differential learning rates for different components
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
        
        # Create optimizer with different learning rates
        self.optimizer = optim.AdamW([
            {'params': text_params, 'lr': float(self.config['training']['text_lr'])},
            {'params': image_params, 'lr': float(self.config['training']['image_lr'])},
            {'params': other_params, 'lr': float(self.config['training']['other_lr'])}
        ], weight_decay=float(self.config['training']['weight_decay']))
        
        # Cosine annealing scheduler with warmup
        total_steps = len(self.train_loader) * int(self.config['training']['epochs'])
        warmup_steps = int(float(self.config['training']['warmup_ratio']) * total_steps)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"✅ Optimizer configured with differential learning rates")
        print(f"📈 Scheduler: Cosine annealing with {warmup_steps} warmup steps")
    
    def train_epoch(self, epoch):
        """Enhanced training epoch with real-time monitoring"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
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
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['max_grad_norm']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })
            
            # Log to tensorboard every 50 steps
            if batch_idx % 50 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Training/Loss_Step', loss.item(), step)
                self.writer.add_scalar('Training/Learning_Rate', current_lr, step)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, epoch):
        """Enhanced validation with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Calculate AUC if binary classification
        if self.config['model']['num_classes'] == 2:
            auc = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
        else:
            auc = 0.0
        
        return avg_loss, accuracy, f1, precision, recall, auc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with comprehensive information"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': float(self.best_val_acc),
            'best_val_f1': float(self.best_val_f1),
            'config': self.config,
            'metrics_history': self.metrics_history
        }
        
        # Save regular checkpoint (use absolute paths)
        checkpoint_dir = BASE_DIR / 'outputs' / 'enhanced_checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, str(checkpoint_path))
        
        # Save best model
        if is_best:
            best_path = BASE_DIR / 'outputs' / 'enhanced_best_model.pth'
            torch.save(checkpoint, str(best_path))
            print(f"💾 New best model saved! Validation Accuracy: {self.best_val_acc:.4f}")
    
    def plot_training_progress(self):
        """Create real-time training progress plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.metrics_history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.metrics_history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(epochs, self.metrics_history['train_f1'], 'b-', label='Train F1')
        axes[1, 0].plot(epochs, self.metrics_history['val_f1'], 'r-', label='Val F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        if self.metrics_history['learning_rates']:
            axes[1, 1].plot(epochs, self.metrics_history['learning_rates'], 'g-')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('outputs/enhanced_training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, csv_path=None, resume_from=None):
        """Main training loop with enhanced monitoring"""
        print("🚀 Starting enhanced training...")
        
        # Check if the CSV file exists
        if csv_path and not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}. Please ensure the file exists at this path. Consider checking the path or placing the file in the project directory.")
        
        # Setup
        train_df, val_df, test_df = self.setup_data(csv_path)
        self.setup_model()
        self.setup_optimizer_scheduler()
        
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            print(f"📂 Resuming from checkpoint: {resume_from}")
            try:
                checkpoint = torch.load(resume_from, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
                self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
                self.metrics_history = checkpoint.get('metrics_history', self.metrics_history)
                print(f"✅ Successfully resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")
                print("🔄 Starting fresh training instead")
                start_epoch = 0
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(start_epoch, self.config['training']['epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.config['training']['epochs']}")
            print("-" * 60)
            
            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_f1, val_precision, val_recall, val_auc = self.validate_epoch(epoch)
            
            # Update metrics history
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['train_acc'].append(train_acc)
            self.metrics_history['val_acc'].append(val_acc)
            self.metrics_history['train_f1'].append(train_f1)
            self.metrics_history['val_f1'].append(val_f1)
            self.metrics_history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            # Log to tensorboard
            self.writer.add_scalar('Training/Loss_Epoch', train_loss, epoch)
            self.writer.add_scalar('Training/Accuracy_Epoch', train_acc, epoch)
            self.writer.add_scalar('Training/F1_Epoch', train_f1, epoch)
            self.writer.add_scalar('Validation/Loss_Epoch', val_loss, epoch)
            self.writer.add_scalar('Validation/Accuracy_Epoch', val_acc, epoch)
            self.writer.add_scalar('Validation/F1_Epoch', val_f1, epoch)
            self.writer.add_scalar('Validation/Precision_Epoch', val_precision, epoch)
            self.writer.add_scalar('Validation/Recall_Epoch', val_recall, epoch)
            if val_auc > 0:
                self.writer.add_scalar('Validation/AUC_Epoch', val_auc, epoch)
            
            # Print metrics
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"Val   - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            if val_auc > 0:
                print(f"Val   - AUC: {val_auc:.4f}")
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot progress
            self.plot_training_progress()
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping']['patience']:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completed
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"🏆 Best validation F1: {self.best_val_f1:.4f}")
        
        # Save final metrics
        with open('outputs/enhanced_training_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.writer.close()
        return self.model, self.metrics_history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run advanced training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config.yaml")
    parser.add_argument("--csv", type=str, default="c:/Users/Shawon/OneDrive/文档/BanglaFakeNewsProject/merged_data.csv", help="Path to merged CSV")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    trainer = AdvancedTrainer()
    if args.epochs is not None:
        # Safely override epochs in config
        trainer.config.setdefault('training', {})
        trainer.config['training']['epochs'] = int(args.epochs)
    model, history = trainer.train(csv_path=args.csv, resume_from=args.resume)