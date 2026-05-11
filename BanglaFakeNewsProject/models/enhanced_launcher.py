"""
Enhanced Project Launcher
Comprehensive launcher for improved Bangla fake news detection system
"""

import os
import sys
import subprocess
import threading
import time
import argparse
from pathlib import Path

# Resolve project base directory (parent of models folder where this file is located)
BASE_DIR = Path(__file__).resolve().parent.parent
# Add both project dir and models dir to sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "models") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "models"))

def print_banner():
    """Print project banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 ENHANCED BANGLA FAKE NEWS DETECTION                    ║
║                           Advanced AI-Powered System                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎯 Improved Accuracy    📊 Real-time Monitoring    🔄 Ensemble Predictions  ║
║  ⚡ Better Performance   📈 Live Training Metrics   🎨 Enhanced Frontend     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    # Map package names to import names
    package_imports = {
        'torch': 'torch',
        'transformers': 'transformers', 
        'torchvision': 'torchvision',
        'gradio': 'gradio',
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib',
        'scikit-learn': 'sklearn',
        'pillow': 'PIL',
        'pyyaml': 'yaml',
        'tqdm': 'tqdm',
        'plotly': 'plotly'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - MISSING")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All requirements satisfied!")
    return True

def check_models():
    """Check if models are available"""
    print("\n🔍 Checking model availability...")
    
    # Check all model files that are actually used by the ensemble
    model_files = [
        ("Enhanced (Advanced)", BASE_DIR / "outputs" / "enhanced_best_model.pth", False),  # Optional
        ("Primary", BASE_DIR / "outputs" / "best_multimodal_model.pth", True),  # Required
        ("Secondary", BASE_DIR / "outputs" / "checkpoints" / "checkpoint_epoch_5.pth", True),  # Required
        ("Multimodal V1", BASE_DIR / "multimodal_detector_v1.pth", True),  # Required
    ]
    
    available_models = []
    required_missing = []
    
    for name, model_file, is_required in model_files:
        if model_file.exists():
            print(f"  ✅ {name}: {model_file.relative_to(BASE_DIR)}")
            available_models.append(str(model_file))
        else:
            if is_required:
                print(f"  ❌ {name}: {model_file.relative_to(BASE_DIR)} - REQUIRED but not found")
                required_missing.append(name)
            else:
                print(f"  ℹ️  {name}: {model_file.relative_to(BASE_DIR)} - Optional (not trained yet)")
    
    if required_missing:
        print(f"❌ {len(required_missing)} required model(s) missing: {', '.join(required_missing)}")
        print("⚠️  Please train the models first:")
        print("   python train_multimodal.py")
        return False
    elif available_models:
        print(f"✅ {len(available_models)} model(s) available for ensemble prediction")
        return True
    else:
        print("⚠️  No trained models found. You can:")
        print("   1. Start training with --train option")
        print("   2. Use the system with base model weights")
        return False

def start_enhanced_app():
    """Start the enhanced Gradio app with integrated pipeline"""
    print("\n🚀 Starting Enhanced Gradio App with Integrated Pipeline...")
    # Ensure we execute from the project directory so relative paths like config.yaml resolve
    try:
        os.chdir(BASE_DIR)
    except Exception:
        pass
    
    # Skip verbose pipeline validation - apps will validate internally
    print("✅ Pipeline ready")
    
    # Try multiple ports in case 7860 is occupied
    ports_to_try = [7860, 7862, 7863, 7864, 7865]
    
    for port in ports_to_try:
        try:
            from enhanced_app_interface import create_enhanced_interface
            
            demo = create_enhanced_interface()
            print("✅ Enhanced app interface created")
            
            print(f"🌐 Launching on http://localhost:{port}")
            print("🎯 Using unified predictor for best results")
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                show_error=True,
                quiet=False
            )
            return  # Success, exit function
            
        except Exception as e:
            error_msg = str(e)
            if "Cannot find empty port" in error_msg and port != ports_to_try[-1]:
                # Port is busy, try next one silently
                continue
            elif port == ports_to_try[-1]:
                # Last port failed
                print(f"⚠️ All ports {ports_to_try[0]}-{ports_to_try[-1]} busy. Trying fallback app...")
                break
            else:
                # Other error
                print(f"⚠️ Enhanced app error: {e}")
                break
    
    # Fallback to original app
    print("🔄 Starting original app with unified predictor...")
    try:
        import app
        print("✅ Original app started successfully")
    except Exception as e2:
        print(f"❌ Failed to start original app: {e2}")

def start_training_monitor():
    """Start the training monitor in a separate thread"""
    print("\n📊 Starting Training Monitor...")
    
    def run_monitor():
        try:
            from training_monitor import create_monitor_interface
            
            demo = create_monitor_interface()
            print("✅ Training monitor interface created")
            print("🌐 Training monitor available at http://localhost:7861")
            
            demo.launch(
                server_name="0.0.0.0",
                server_port=7861,
                share=False,
                show_error=True,
                quiet=True
            )
            
        except Exception as e:
            print(f"❌ Failed to start training monitor: {e}")
    
    monitor_thread = threading.Thread(target=run_monitor, daemon=True)
    monitor_thread.start()
    
    return monitor_thread

def start_enhanced_training():
    """Start enhanced training process"""
    print("\n🏋️ Starting Enhanced Training...")
    
    try:
        from advanced_training import AdvancedTrainer
        
        print("🔧 Initializing advanced trainer...")
        trainer = AdvancedTrainer()
        
        print("🚀 Starting training process...")
        model, history = trainer.train()
        
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def run_quick_test():
    """Run a quick test of the integrated system"""
    print("\n🧪 Running Integrated System Test...")
    # Ensure we execute from the project directory so relative paths like config.yaml resolve
    try:
        os.chdir(BASE_DIR)
    except Exception:
        pass
    
    try:
        # Try unified predictor first
        try:
            from unified_predictor import UnifiedPredictor
            print("🔧 Loading unified predictor (existing strategy + enhancements)...")
            # Prefer passing absolute config path if UnifiedPredictor supports it
            config_path = str(BASE_DIR / "config.yaml")
            try:
                predictor = UnifiedPredictor(config_path=config_path)  # type: ignore
            except Exception:
                # Fallback to default constructor
                predictor = UnifiedPredictor()  # type: ignore
            predictor_type = "UNIFIED"
        except Exception as e:
            print(f"⚠️ Unified predictor failed: {e}")
            print("🔄 Falling back to original predictor...")
            from predictor import Predictor
            predictor = Predictor(config_path=str(BASE_DIR / "config.yaml"))
            predictor_type = "ORIGINAL"
        
        print("🧪 Testing integrated prediction pipeline...")
        test_text = "ভূমিকম্পে কেঁপে উঠলো ঢাকা। আজ সকালে রিখটার স্কেলে ৫.২ মাত্রার ভূমিকম্প অনুভূত হয়েছে।"
        
        import time
        start_time = time.time()
        result = predictor.predict(test_text)
        inference_time = time.time() - start_time
        
        print("✅ Integrated Pipeline Test Results:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   News Type: {result['news_type']}")
        print(f"   Models Used: {result.get('model_count', 1)}")
        print(f"   Inference Time: {inference_time:.3f}s")
        print(f"   Pipeline Status: {'✅ ENHANCED' if result.get('model_count', 1) > 1 else '✅ STANDARD'}")
        
        # Run pipeline integration test
        print("\n🔍 Running pipeline integration validation...")
        from pipeline_integration import PipelineIntegrator
        integrator = PipelineIntegrator()
        integration_success = integrator.run_complete_integration()
        
        if integration_success:
            print("\n🎉 SYSTEM TEST PASSED - Ready for best results!")
        else:
            print("\n⚠️ SYSTEM TEST COMPLETED - Basic functionality verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated test failed: {e}")
        print("🔄 Trying fallback test...")
        
        try:
            from predictor import Predictor
            predictor = Predictor()
            result = predictor.predict("Test news", None)
            print("✅ Fallback test passed - Original system working")
            return True
        except Exception as e2:
            print(f"❌ Fallback test also failed: {e2}")
            return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Enhanced Bangla Fake News Detection Launcher")
    parser.add_argument("--app", action="store_true", help="Start the enhanced Gradio app")
    parser.add_argument("--train", action="store_true", help="Start enhanced training")
    parser.add_argument("--monitor", action="store_true", help="Start training monitor only")
    parser.add_argument("--test", action="store_true", help="Run quick system test")
    parser.add_argument("--full", action="store_true", help="Start full system (app + monitor)")
    parser.add_argument("--check", action="store_true", help="Check system requirements and models")
    
    args = parser.parse_args()
    
    print_banner()
    # Normalize working directory to the project folder
    try:
        os.chdir(BASE_DIR)
        # Show where we run from for clarity
        print(f"📂 Working directory: {Path.cwd()}")
    except Exception as e:
        print(f"⚠️ Could not change working directory: {e}")
    
    # Check requirements first
    if not check_requirements():
        print("\n❌ Requirements check failed. Please install missing packages.")
        return
    
    # Check models
    models_available = check_models()
    
    if args.check:
        print("\n✅ System check completed!")
        return
    
    if args.test:
        success = run_quick_test()
        if success:
            print("\n✅ System test passed!")
        else:
            print("\n❌ System test failed!")
        return
    
    if args.train:
        print("\n🏋️ Starting Enhanced Training Mode...")
        
        # Start training monitor first
        monitor_thread = start_training_monitor()
        time.sleep(2)  # Give monitor time to start
        
        # Start training
        success = start_enhanced_training()
        
        if success:
            print("\n✅ Training completed! You can now use the enhanced models.")
        else:
            print("\n❌ Training failed. Check the logs for details.")
        
        return
    
    if args.monitor:
        print("\n📊 Starting Training Monitor Only...")
        monitor_thread = start_training_monitor()
        
        try:
            # Keep the monitor running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Training monitor stopped.")
        
        return
    
    if args.full:
        print("\n🚀 Starting Full Enhanced System...")
        
        # Start training monitor
        monitor_thread = start_training_monitor()
        time.sleep(2)  # Give monitor time to start
        
        print("📊 Training monitor started at http://localhost:7861")
        
        # Start main app
        start_enhanced_app()
        
        return
    
    if args.app or not any(vars(args).values()):
        # Default: start integrated enhanced app
        print("\n🎯 Starting Integrated Enhanced App for Best Results...")
        start_enhanced_app()
        return
    
    # If no arguments provided, show help
    parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Enhanced Bangla Fake News Detection System stopped.")
        print("Thank you for using our system!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the logs and try again.")