#!/usr/bin/env python3
"""
Comprehensive Data Processing and Fake News Detection Script
Processes real data from CSV files and runs detection on multiple samples
"""

import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
import random
from pathlib import Path

# Add current directory to path to import the model
sys.path.append('.')

from model_defs import OptimizedMultimodalModel
from transformers import AutoTokenizer

class BanglaFakeNewsDetector:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.image_transform = None
        
    def load_model(self):
        """Load the model and tokenizer"""
        print("🔄 Loading model and tokenizer...")
        
        # Initialize model
        self.model = OptimizedMultimodalModel()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer with offline mode fallback
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert", local_files_only=True)
        except Exception as e:
            print(f"⚠️ Loading from cache failed, trying online (may fail if offline): {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
        
        # Image transform
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ Model and tokenizer loaded successfully!")
        
    def load_data(self, csv_path, sample_size=10):
        """Load and sample data from CSV file"""
        print(f"🔄 Loading data from {csv_path}...")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} records from {csv_path}")
            
            # Sample data
            if len(df) > sample_size:
                df_sample = df.sample(n=sample_size, random_state=42)
            else:
                df_sample = df
                
            print(f"📊 Processing {len(df_sample)} samples")
            return df_sample
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def predict_single(self, text, image_path):
        """Make prediction on a single sample"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Load and transform image
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.image_transform(image).unsqueeze(0)
            else:
                # Create dummy image if not found
                image_tensor = torch.randn(1, 3, 224, 224)
                print(f"⚠️  Image {image_path} not found, using dummy image")
            
            # Move to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            image_tensor = image_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1)
            
            return {
                'prediction': prediction.item(),
                'probabilities': probabilities.cpu().numpy()[0],
                'confidence': torch.max(probabilities).item(),
                'pred_label': "Fake" if prediction.item() == 1 else "Real"
            }
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return None
    
    def process_dataset(self, df, image_dir="images"):
        """Process entire dataset and return results"""
        results = []
        
        print(f"\n🔍 Processing {len(df)} samples...")
        print("="*80)
        
        for idx, row in df.iterrows():
            # Get text (use headline as primary text)
            text = str(row.get('headline', ''))
            if pd.isna(text) or text == '':
                text = str(row.get('description', ''))
            
            # Get image path
            image_id = str(row.get('image_id', ''))
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            
            # Get true label
            true_label = int(row.get('label', 0))
            true_label_text = "Fake" if true_label == 1 else "Real"
            
            print(f"\n📰 Sample {idx + 1}/{len(df)}")
            print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"Image: {image_path}")
            print(f"True Label: {true_label_text}")
            
            # Make prediction
            prediction_result = self.predict_single(text, image_path)
            
            if prediction_result:
                pred_label = prediction_result['pred_label']
                confidence = prediction_result['confidence']
                prob_real = prediction_result['probabilities'][0]
                prob_fake = prediction_result['probabilities'][1]
                
                # Check if prediction is correct
                is_correct = (prediction_result['prediction'] == true_label)
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                
                print(f"Prediction: {pred_label}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Probabilities - Real: {prob_real:.4f}, Fake: {prob_fake:.4f}")
                print(f"Status: {status}")
                
                # Store results
                results.append({
                    'sample_id': idx + 1,
                    'text': text,
                    'image_path': image_path,
                    'true_label': true_label,
                    'true_label_text': true_label_text,
                    'predicted_label': prediction_result['prediction'],
                    'predicted_label_text': pred_label,
                    'confidence': confidence,
                    'prob_real': prob_real,
                    'prob_fake': prob_fake,
                    'is_correct': is_correct
                })
            else:
                print("❌ Prediction failed")
                results.append({
                    'sample_id': idx + 1,
                    'text': text,
                    'image_path': image_path,
                    'true_label': true_label,
                    'true_label_text': true_label_text,
                    'predicted_label': -1,
                    'predicted_label_text': "ERROR",
                    'confidence': 0.0,
                    'prob_real': 0.0,
                    'prob_fake': 0.0,
                    'is_correct': False
                })
        
        return results
    
    def analyze_results(self, results):
        """Analyze and display results"""
        if not results:
            print("❌ No results to analyze")
            return
        
        print("\n" + "="*80)
        print("📊 RESULTS ANALYSIS")
        print("="*80)
        
        # Calculate metrics
        total_samples = len(results)
        correct_predictions = sum(1 for r in results if r['is_correct'])
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        # Count by true labels
        real_samples = [r for r in results if r['true_label'] == 0]
        fake_samples = [r for r in results if r['true_label'] == 1]
        
        real_correct = sum(1 for r in real_samples if r['is_correct'])
        fake_correct = sum(1 for r in fake_samples if r['is_correct'])
        
        real_accuracy = real_correct / len(real_samples) if real_samples else 0
        fake_accuracy = fake_correct / len(fake_samples) if fake_samples else 0
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in results if r['confidence'] > 0])
        
        print(f"📈 Overall Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
        print(f"📈 Real News Accuracy: {real_accuracy:.4f} ({real_correct}/{len(real_samples)})")
        print(f"📈 Fake News Accuracy: {fake_accuracy:.4f} ({fake_correct}/{len(fake_samples)})")
        print(f"📈 Average Confidence: {avg_confidence:.4f}")
        
        # Show detailed breakdown
        print(f"\n📋 Detailed Breakdown:")
        print(f"   Total Samples: {total_samples}")
        print(f"   Real News Samples: {len(real_samples)}")
        print(f"   Fake News Samples: {len(fake_samples)}")
        print(f"   Correct Predictions: {correct_predictions}")
        print(f"   Wrong Predictions: {total_samples - correct_predictions}")
        
        # Show some examples
        print(f"\n🎯 Sample Results:")
        for i, result in enumerate(results[:5]):  # Show first 5 results
            status = "✅" if result['is_correct'] else "❌"
            print(f"   {i+1}. {status} True: {result['true_label_text']} | Pred: {result['predicted_label_text']} | Conf: {result['confidence']:.3f}")

def main():
    print("="*80)
    print("🔍 BANGLA FAKE NEWS DETECTION - DATA PROCESSING & DETECTION")
    print("="*80)
    
    # Initialize detector
    detector = BanglaFakeNewsDetector()
    detector.load_model()
    
    # Process different datasets - FIXED PATHS
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = [
        (os.path.join(base_dir, "Test.csv"), 15),  # Test set with 15 samples
        (os.path.join(base_dir, "Train.csv"), 10), # Train set with 10 samples
    ]
    
    all_results = []
    
    for csv_file, sample_size in datasets:
        if os.path.exists(csv_file):
            print(f"\n{'='*80}")
            print(f"📁 PROCESSING {csv_file.upper()}")
            print(f"{'='*80}")
            
            # Load data
            df = detector.load_data(csv_file, sample_size)
            if df is not None:
                # Process dataset
                results = detector.process_dataset(df)
                all_results.extend(results)
                
                # Analyze results for this dataset
                detector.analyze_results(results)
        else:
            print(f"❌ File {csv_file} not found")
    
    # Overall analysis
    if all_results:
        print(f"\n{'='*80}")
        print("🎯 OVERALL RESULTS SUMMARY")
        print(f"{'='*80}")
        detector.analyze_results(all_results)
        
        # Save results to file
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('detection_results.csv', index=False)
        print(f"\n💾 Results saved to 'detection_results.csv'")
    
    print(f"\n{'='*80}")
    print("✅ DATA PROCESSING AND DETECTION COMPLETED!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()