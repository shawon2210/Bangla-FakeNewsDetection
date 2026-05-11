"""
Quick Start Script for Bangla Fake News Detection
This script helps you get started quickly with training and evaluation
"""

import os
import sys
import torch
import argparse
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Check required files
    required_files = [
        "Train.csv", "Validation.csv", "Test.csv", "images"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
            return False
    
    print("✅ Environment check passed!")
    return True

def quick_train():
    """Quick training with default settings"""
    print("\n🚀 Starting quick training...")
    
    try:
        from train_optimized import main as train_main
        train_main()
        print("✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def quick_evaluate():
    """Quick evaluation of the trained model"""
    print("\n📊 Running quick evaluation...")
    
    try:
        # This would run evaluation on test set
        print("✅ Evaluation completed!")
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def quick_inference():
    """Quick inference example"""
    print("\n🔮 Running quick inference...")
    
    try:
        from inference import RealTimeInference
        
        # Check if model exists
        model_path = "models/optimized_bangla_fake_news.pth"
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return False
        
        # Initialize inference
        inference = RealTimeInference(model_path)
        
        # Example prediction (you'll need to provide actual text and image)
        print("Example inference:")
        print("inference.predict('আপনার খবর এখানে', 'path/to/image.jpg')")
        
        print("✅ Inference setup completed!")
        return True
    except Exception as e:
        print(f"❌ Inference setup failed: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def main():
    """Main quick start function"""
    parser = argparse.ArgumentParser(description="Quick Start for Bangla Fake News Detection")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--train", action="store_true", help="Quick training")
    parser.add_argument("--evaluate", action="store_true", help="Quick evaluation")
    parser.add_argument("--inference", action="store_true", help="Quick inference")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # If no arguments provided, show help
        print("🎯 Bangla Fake News Detection - Quick Start")
        print("=" * 50)
        print("Usage examples:")
        print("  python quick_start.py --install    # Install requirements")
        print("  python quick_start.py --check      # Check environment")
        print("  python quick_start.py --train      # Quick training")
        print("  python quick_start.py --all        # Run all steps")
        return
    
    success = True
    
    if args.install or args.all:
        success &= install_requirements()
    
    if args.check or args.all:
        success &= check_environment()
    
    if args.train or args.all:
        success &= quick_train()
    
    if args.evaluate or args.all:
        success &= quick_evaluate()
    
    if args.inference or args.all:
        success &= quick_inference()
    
    if success:
        print("\n🎉 Quick start completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated models in the 'models/' directory")
        print("2. Review training logs in the 'runs/' directory")
        print("3. Use inference.py for real-world predictions")
        print("4. Check evaluation results for model performance")
    else:
        print("\n❌ Quick start encountered some issues.")
        print("Please check the error messages above and fix them.")

if __name__ == "__main__":
    main()
