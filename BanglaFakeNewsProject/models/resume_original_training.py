"""
Resume Original Training - Continue your existing training approach
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_multimodal import train_model

def find_latest_checkpoint():
    """Find the most recent checkpoint"""
    checkpoint_paths = [
        "../outputs/checkpoints/checkpoint_epoch_5.pth",
        "../checkpoints/checkpoint_epoch_5.pth",
        "../multimodal_detector_v1.pth"
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✅ Found checkpoint: {path}")
            return path
    
    print("⚠️ No checkpoint found")
    return None

def main():
    """Resume training with your existing approach"""
    print("🔄 Resuming Original Training Approach...")
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    
    # Resume training using your existing function
    print("🚀 Starting training...")
    
    try:
        # Use your existing training function with resume capability
        model, history = train_model(
            csv_path="../merged_data.csv",  # Use merged data
            pretrained_path=checkpoint_path,  # Resume from checkpoint
            num_epochs=15,  # Continue to 15 epochs
            batch_size=8,   # Optimized batch size
            lr=2e-5,        # Optimized learning rate
            num_workers=0   # Windows compatibility
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Check if all dependencies are installed and models are available")

if __name__ == "__main__":
    main()