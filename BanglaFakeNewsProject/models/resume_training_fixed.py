"""
Fixed Resume Training Script
"""

import os
import sys
import torch
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_multimodal import train_model

def main():
    print("🔄 Resuming Training with Fixed Settings...")
    
    # Find latest checkpoint
    checkpoint_path = "../outputs/checkpoints/checkpoint_epoch_5.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Found checkpoint: {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found, starting fresh")
        checkpoint_path = None
    
    try:
        # Resume training with your existing function
        print("🚀 Starting training...")
        
        model, history = train_model(
            csv_path="../merged_data.csv",
            text_col='headline',
            image_col='image_id', 
            label_col='label',
            batch_size=8,
            max_text_length=128,
            num_epochs=15,
            lr=2e-5,
            weight_decay=0.01,
            early_stopping_patience=10,
            mixed_precision=False,
            accumulation_steps=1,
            num_workers=0,  # Windows compatibility
            pretrained_path=checkpoint_path
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()