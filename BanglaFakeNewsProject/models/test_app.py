"""
Test App - Simple test to verify everything works
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_predictor():
    """Test predictor functionality"""
    print("🧪 Testing predictor...")
    
    try:
        # Try unified predictor first
        try:
            from unified_predictor import UnifiedPredictor
            predictor = UnifiedPredictor()
            print("✅ Unified predictor loaded")
        except Exception as e:
            print(f"⚠️ Unified predictor failed: {e}")
            # Fallback to original
            from predictor import Predictor
            predictor = Predictor()
            print("✅ Original predictor loaded")
        
        # Test prediction
        test_text = "Test news article"
        result = predictor.predict(test_text, None)
        
        print(f"✅ Prediction: {result['prediction']}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        print(f"✅ News Type: {result['news_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_app():
    """Test app loading"""
    print("🧪 Testing app...")
    
    try:
        # Change to parent directory
        os.chdir(parent_dir)
        
        # Import app
        import app
        print("✅ App imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ App test failed: {e}")
        return False

def main():
    print("🚀 Running Tests...")
    print("=" * 40)
    
    predictor_ok = test_predictor()
    app_ok = test_app()
    
    if predictor_ok and app_ok:
        print("\n✅ All tests passed! System is working.")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")

if __name__ == "__main__":
    main()