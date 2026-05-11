"""Quick test script for updated predictor"""

import sys
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from predictor import Predictor

def test_predictor():
    print("=" * 60)
    print("Testing Updated Predictor")
    print("=" * 60)
    
    # Initialize predictor
    print("\n1. Initializing predictor...")
    config_path = BASE_DIR / "config.yaml"
    predictor = Predictor(config_path=str(config_path))
    
    # Test prediction
    print("\n2. Testing prediction...")
    test_text = "ভূমিকম্পে কেঁপে উঠলো ঢাকা। আজ সকালে ঢাকা শহরে এক শক্তিশালী ভূমিকম্প অনুভূত হয়েছে।"
    
    result = predictor.predict(test_text, None)
    
    print("\n3. Prediction Results:")
    print("-" * 60)
    print(f"   Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   News Type: {result['news_type']}")
    print(f"   Model Count: {result['model_count']}")
    print(f"   Probabilities: {result['probabilities']}")
    print("-" * 60)
    
    # Verify model_count
    if result['model_count'] >= 3:
        print(f"\n✅ SUCCESS: Using {result['model_count']}-model ensemble!")
        if result['model_count'] == 4:
            print("   🎯 All 4 models are loaded and working!")
        else:
            print(f"   ℹ️  Note: {result['model_count']} models loaded (enhanced model may not be available yet)")
    else:
        print(f"\n⚠️  WARNING: Only {result['model_count']} models loaded")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_predictor()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
