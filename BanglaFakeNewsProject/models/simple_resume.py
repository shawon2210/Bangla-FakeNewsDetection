"""
Simple Resume Training - Direct CSV approach
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_multimodal import train_model

def main():
    print("🔄 Simple Resume Training...")
    
    # Use merged_data.csv directly (your existing approach)
    csv_path = "../merged_data.csv"
    checkpoint_path = "../outputs/checkpoints/checkpoint_epoch_5.pth"
    
    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found")
        return
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Found checkpoint: {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found, starting fresh")
        checkpoint_path = None
    
    try:
        print("🚀 Starting training...")
        
        # Use your exact working approach
        model, history = train_model(
            csv_path=csv_path,  # Use merged_data.csv
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
            num_workers=0,
            pretrained_path=checkpoint_path
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()