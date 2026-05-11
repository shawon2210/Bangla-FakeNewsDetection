"""
Quick Test Script - Simple validation without complex dependencies
"""

def test_basic_imports():
    """Test basic package imports"""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print("✅ torch")
    except:
        print("❌ torch")
        return False
    
    try:
        import yaml
        print("✅ yaml (pyyaml)")
    except:
        print("❌ yaml (pyyaml)")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL (pillow)")
    except:
        print("❌ PIL (pillow)")
        return False
    
    try:
        import sklearn
        print("✅ sklearn (scikit-learn)")
    except:
        print("❌ sklearn (scikit-learn)")
        return False
    
    return True

def test_predictor():
    """Test predictor functionality"""
    print("\n🧪 Testing predictor...")
    
    try:
        from predictor import Predictor
        predictor = Predictor()
        
        test_text = "Test news article"
        result = predictor.predict(test_text, None)
        
        print(f"✅ Prediction: {result['prediction']}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Predictor test failed: {e}")
        return False

def main():
    print("🚀 Quick System Test")
    print("=" * 40)
    
    if not test_basic_imports():
        print("\n❌ Basic imports failed. Please install missing packages:")
        print("pip install scikit-learn pillow pyyaml")
        return
    
    if not test_predictor():
        print("\n❌ Predictor test failed. Check model files.")
        return
    
    print("\n✅ Quick test passed! System is working.")

if __name__ == "__main__":
    main()