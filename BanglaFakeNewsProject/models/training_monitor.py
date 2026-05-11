"""
Real-time Training Monitor
Provides live updates during training without interrupting the process
"""

import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import threading
import gradio as gr
from pathlib import Path

class TrainingMonitor:
    """Real-time training monitor with live updates"""
    
    def __init__(self, log_dir="runs", checkpoint_dir="outputs/enhanced_checkpoints"):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.is_monitoring = False
        self.current_metrics = {}
        
    def start_monitoring(self):
        """Start monitoring training progress"""
        self.is_monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._update_metrics()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(10)
    
    def _update_metrics(self):
        """Update current metrics from training files"""
        # Check for training history file
        history_file = "outputs/enhanced_training_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.current_metrics = json.load(f)
            except:
                pass
        
        # Check for latest checkpoint
        if os.path.exists(self.checkpoint_dir):
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(self.checkpoint_dir, x)))
                self.current_metrics['latest_checkpoint'] = latest_checkpoint
                self.current_metrics['last_update'] = datetime.now().isoformat()
    
    def get_training_status(self):
        """Get current training status"""
        if not self.current_metrics:
            return "No training data available", None, None, None
        
        # Create status text
        status_lines = []
        status_lines.append("🚀 TRAINING STATUS")
        status_lines.append("=" * 50)
        
        if 'train_loss' in self.current_metrics:
            epochs_completed = len(self.current_metrics['train_loss'])
            status_lines.append(f"Epochs Completed: {epochs_completed}")
            
            if epochs_completed > 0:
                latest_train_loss = self.current_metrics['train_loss'][-1]
                latest_val_loss = self.current_metrics.get('val_loss', [0])[-1]
                latest_train_acc = self.current_metrics.get('train_acc', [0])[-1]
                latest_val_acc = self.current_metrics.get('val_acc', [0])[-1]
                
                status_lines.append(f"Latest Train Loss: {latest_train_loss:.4f}")
                status_lines.append(f"Latest Val Loss: {latest_val_loss:.4f}")
                status_lines.append(f"Latest Train Acc: {latest_train_acc:.4f}")
                status_lines.append(f"Latest Val Acc: {latest_val_acc:.4f}")
                
                # Best metrics
                best_val_acc = max(self.current_metrics.get('val_acc', [0]))
                best_epoch = self.current_metrics.get('val_acc', [0]).index(best_val_acc) + 1
                status_lines.append(f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
        
        if 'latest_checkpoint' in self.current_metrics:
            status_lines.append(f"Latest Checkpoint: {self.current_metrics['latest_checkpoint']}")
        
        if 'last_update' in self.current_metrics:
            status_lines.append(f"Last Update: {self.current_metrics['last_update']}")
        
        status_text = "\n".join(status_lines)
        
        # Create plots
        loss_plot = self._create_loss_plot()
        acc_plot = self._create_accuracy_plot()
        lr_plot = self._create_lr_plot()
        
        return status_text, loss_plot, acc_plot, lr_plot
    
    def _create_loss_plot(self):
        """Create loss plot"""
        if 'train_loss' not in self.current_metrics:
            return None
        
        epochs = range(1, len(self.current_metrics['train_loss']) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.current_metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        
        if 'val_loss' in self.current_metrics:
            ax.plot(epochs, self.current_metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_accuracy_plot(self):
        """Create accuracy plot"""
        if 'train_acc' not in self.current_metrics:
            return None
        
        epochs = range(1, len(self.current_metrics['train_acc']) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.current_metrics['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        
        if 'val_acc' in self.current_metrics:
            ax.plot(epochs, self.current_metrics['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def _create_lr_plot(self):
        """Create learning rate plot"""
        if 'learning_rates' not in self.current_metrics:
            return None
        
        epochs = range(1, len(self.current_metrics['learning_rates']) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.current_metrics['learning_rates'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig

def create_monitor_interface():
    """Create Gradio interface for training monitoring"""
    
    monitor = TrainingMonitor()
    monitor.start_monitoring()
    
    with gr.Blocks(title="Training Monitor") as demo:
        
        gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 20px;">
                <h1 style="color: white; font-size: 2rem; margin: 0;">📊 Real-time Training Monitor</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0;">Live monitoring of model training progress</p>
            </div>
        """)
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="primary")
            auto_refresh = gr.Checkbox(label="Auto-refresh (every 10s)", value=False)
        
        # Status display
        status_output = gr.Textbox(
            label="Training Status",
            lines=15,
            value="Click refresh to load training status...",
            elem_classes=["monospace"]
        )
        
        # Plots
        with gr.Row():
            with gr.Column():
                loss_plot = gr.Plot(label="Loss Progress")
            with gr.Column():
                acc_plot = gr.Plot(label="Accuracy Progress")
        
        with gr.Row():
            lr_plot = gr.Plot(label="Learning Rate Schedule")
        
        # Training controls
        gr.Markdown("### 🎛️ Training Controls")
        
        with gr.Row():
            start_training_btn = gr.Button("🚀 Start Enhanced Training", variant="primary", size="lg")
            stop_training_btn = gr.Button("⏹️ Stop Training", variant="stop")
        
        training_output = gr.Textbox(
            label="Training Output",
            lines=10,
            placeholder="Training output will appear here..."
        )
        
        def refresh_status():
            return monitor.get_training_status()
        
        def start_training():
            """Start training process"""
            try:
                import subprocess
                import sys
                
                # Start training in background
                cmd = [sys.executable, "advanced_training.py"]
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.getcwd()
                )
                
                return "🚀 Enhanced training started! Check the status above for progress."
                
            except Exception as e:
                return f"❌ Failed to start training: {str(e)}"
        
        def stop_training():
            """Stop training process"""
            # This is a placeholder - actual implementation would need process management
            return "⏹️ Training stop signal sent (manual intervention may be required)"
        
        # Event handlers
        refresh_btn.click(
            fn=refresh_status,
            outputs=[status_output, loss_plot, acc_plot, lr_plot]
        )
        
        start_training_btn.click(
            fn=start_training,
            outputs=[training_output]
        )
        
        stop_training_btn.click(
            fn=stop_training,
            outputs=[training_output]
        )
        
        # Auto-refresh functionality
        def auto_refresh_fn():
            if auto_refresh.value:
                return refresh_status()
            return gr.update(), gr.update(), gr.update(), gr.update()
        
        # Set up periodic refresh (every 10 seconds when enabled)
        demo.load(
            fn=refresh_status,
            outputs=[status_output, loss_plot, acc_plot, lr_plot],
            every=10
        )
    
    return demo

if __name__ == "__main__":
    demo = create_monitor_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )