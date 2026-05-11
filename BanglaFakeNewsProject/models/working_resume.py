"""
Working Resume Script - Guaranteed to work from models directory
"""

import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Change to parent directory
os.chdir(parent_dir)

def main():
    print("🔄 Working Resume Training...")
    print(f"📁 Working directory: {os.getcwd()}")
    
    try:
        # Import after path setup
        from train_multimodal import train_model
        
        # Check for checkpoint
        checkpoint_path = "outputs/checkpoints/checkpoint_epoch_5.pth"
        if os.path.exists(checkpoint_path):
            print(f"✅ Found checkpoint: {checkpoint_path}")
        else:
            print("⚠️ No checkpoint found, starting fresh")
            checkpoint_path = None
        
        # Check for data
        csv_path = "merged_data.csv"
        if not os.path.exists(csv_path):
            print(f"❌ {csv_path} not found")
            return
        
        print("🚀 Starting training...")
        
        # Resume training
        model, history = train_model(
            csv_path=csv_path,
            text_col='headline',
            image_col='image_id', 
            label_col='label',
            batch_size=8,
            max_text_length=128,
            num_epochs=10,
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()