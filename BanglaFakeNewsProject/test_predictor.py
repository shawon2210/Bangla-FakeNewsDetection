import os
import sys
import torch

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import Predictor


# Use a real image from the dataset for testing
test_image_path = os.path.join(os.path.dirname(__file__), "images", "politics_fake_236.jfif")

# Initialize the predictor
print("Initializing Predictor...")
predictor = Predictor(config_path="C:\\Users\\Shawon\\OneDrive\\文档\\BanglaFakeNewsProject\\BanglaFakeNewsProject\\config.yaml")
print("Predictor initialized.")


# Test with sample data (text + image)
sample_text = "ভূমিকম্পে কেঁপে উঠলো ঢাকা"

print(f"\nMaking prediction for text: '{sample_text}' and image: '{test_image_path}'")
result = predictor.predict(sample_text, test_image_path)

print("\nPrediction Result:")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"News Type: {result['news_type']}")
print(f"Probabilities: {result['probabilities']}")

print("\nTest complete.")

