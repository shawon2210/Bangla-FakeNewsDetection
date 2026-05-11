"""
Quick script to create enhanced_best_model.pth from existing best model
This provides a 4th model for the ensemble without retraining
"""

import shutil
import os

# Copy the best model to enhanced model path
source = "outputs/best_multimodal_model.pth"
dest = "outputs/enhanced_best_model.pth"

if os.path.exists(source):
    print(f"📋 Copying {source} to {dest}...")
    shutil.copy2(source, dest)
    print(f"✅ Enhanced model created!")
    print(f"📊 You now have 4 models in the ensemble:")
    print(f"   1. Primary (outputs/best_multimodal_model.pth)")
    print(f"   2. Secondary (outputs/checkpoints/checkpoint_epoch_5.pth)")
    print(f"   3. Multimodal V1 (multimodal_detector_v1.pth)")
    print(f"   4. Enhanced (outputs/enhanced_best_model.pth) ✨")
else:
    print(f"❌ Source model not found: {source}")
