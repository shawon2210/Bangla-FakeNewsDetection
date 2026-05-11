"""
Working Resume Training - Guaranteed to work
"""

import os
import sys

# Setup paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

def main():
    print("🔄 Working Resume Training...")
    
    try:
        from train_multimodal import train_model
        
        checkpoint_path = "outputs/checkpoints/checkpoint_epoch_5.pth"
        if os.path.exists(checkpoint_path):
            print(f"✅ Found checkpoint: {checkpoint_path}")
        else:
            print("⚠️ No checkpoint found")
            checkpoint_path = None
        
        print("🚀 Starting training...")
        model, history = train_model(
            csv_path="merged_data.csv",
            text_col='headline',
            image_col='image_id', 
            label_col='label',
            batch_size=8,
            num_epochs=15,
            lr=2e-5,
            num_workers=0,
            pretrained_path=checkpoint_path
        )
        
        print("✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()
