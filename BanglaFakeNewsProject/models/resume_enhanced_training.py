"""
Resume Enhanced Training - Continue from latest checkpoint
"""

import os
import sys
import torch
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_training import AdvancedTrainer

def find_latest_checkpoint():
    """Find the most recent checkpoint"""
    checkpoint_dirs = [
        "../outputs/checkpoints",
        "../checkpoints", 
        "../outputs/enhanced_checkpoints"
    ]
    
    latest_checkpoint = None
    latest_epoch = -1
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pth') and 'checkpoint_epoch_' in file:
                    try:
                        epoch_num = int(file.split('_')[-1].split('.')[0])
                        if epoch_num > latest_epoch:
                            latest_epoch = epoch_num
                            latest_checkpoint = os.path.join(checkpoint_dir, file)
                    except:
                        continue
    
    return latest_checkpoint, latest_epoch

def resume_training():
    """Resume training from latest checkpoint"""
    print("🔍 Looking for latest checkpoint...")
    
    latest_checkpoint, latest_epoch = find_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"✅ Found checkpoint: {latest_checkpoint} (Epoch {latest_epoch})")
        print(f"🔄 Resuming training from epoch {latest_epoch + 1}")
    else:
        print("⚠️ No checkpoint found, starting fresh training")
        latest_checkpoint = None
    
    # Initialize trainer with correct config path
    trainer = AdvancedTrainer("../config.yaml")
    
    # Start/resume training
    print("🚀 Starting enhanced training...")
    model, history = trainer.train(resume_from=latest_checkpoint)
    
    print("✅ Training completed!")
    return model, history

if __name__ == "__main__":
    try:
        model, history = resume_training()
        print("🎉 Enhanced training finished successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try running the original training script if enhanced training fails")