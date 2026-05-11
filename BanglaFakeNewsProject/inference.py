"""
Real-world Inference Script for Bangla Fake News Detection
Optimized for production deployment with batch processing
"""

import os
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import argparse
from pathlib import Path

from model_defs import OptimizedMultimodalModel, ModelManager

class InferenceDataset(Dataset):
    """Dataset class for inference"""
    
    def __init__(self, data, tokenizer, image_transform, image_folder="images"):
        self.data = data
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_folder = image_folder
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process text
        text = item['text']
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Process image
        image_path = item['image_path']
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image)
        else:
            # Create dummy image if not found
            image_tensor = torch.zeros(3, 224, 224)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image_tensor,
            'text': text,
            'image_path': image_path,
            'id': item.get('id', idx)
        }

class BatchInference:
    """Batch inference for efficient processing"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_manager = ModelManager(model_path, self.device)
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single(self, text: str, image_path: str) -> Dict:
        """Predict single sample"""
        try:
            result = self.model_manager.predict(text, image_path, self.image_transform)
            return {
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'].tolist(),
                'text': text,
                'image_path': image_path
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': text,
                'image_path': image_path
            }
    
    def predict_batch(self, data: List[Dict], batch_size: int = 16) -> List[Dict]:
        """Predict batch of samples"""
        results = []
        
        # Create dataset and dataloader
        dataset = InferenceDataset(
            data, 
            self.model_manager.tokenizer, 
            self.image_transform
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model_manager.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                
                # Predict
                outputs = self.model_manager.model(input_ids, attention_mask, images)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                # Process results
                for i in range(len(batch['text'])):
                    result = {
                        'success': True,
                        'id': batch['id'][i],
                        'text': batch['text'][i],
                        'image_path': batch['image_path'][i],
                        'prediction': predictions[i].item(),
                        'confidence': confidences[i].item(),
                        'probabilities': probabilities[i].cpu().numpy().tolist()
                    }
                    results.append(result)
        
        return results

class RealTimeInference:
    """Real-time inference for web applications"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_manager = ModelManager(model_path, self.device)
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Performance tracking
        self.inference_times = []
    
    def predict(self, text: str, image_path: str) -> Dict:
        """Real-time prediction with performance tracking"""
        start_time = time.time()
        
        try:
            # Predict
            result = self.model_manager.predict(text, image_path, self.image_transform)
            
            # Track inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return {
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'].tolist(),
                'inference_time': inference_time,
                'label': 'Fake' if result['prediction'] == 1 else 'Real'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {'message': 'No inference data available'}
        
        return {
            'total_inferences': len(self.inference_times),
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times)
        }

def load_data_from_csv(csv_path: str, text_column: str = 'headline', 
                      image_id_column: str = 'image_id', 
                      image_folder: str = 'images') -> List[Dict]:
    """Load data from CSV file"""
    df = pd.read_csv(csv_path)
    data = []
    
    for idx, row in df.iterrows():
        # Try different image extensions
        image_id = row[image_id_column]
        image_path = None
        
        for ext in ['.jpg', '.png', '.jpeg', '.webp', '.gif', '.jfif']:
            potential_path = os.path.join(image_folder, image_id + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"Warning: Image not found for ID: {image_id}")
            continue
        
        data.append({
            'id': idx,
            'text': str(row[text_column]),
            'image_path': image_path
        })
    
    return data

def save_results(results: List[Dict], output_path: str):
    """Save prediction results"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

def evaluate_model_performance(results: List[Dict], ground_truth_path: str = None):
    """Evaluate model performance if ground truth is available"""
    if ground_truth_path is None:
        print("No ground truth provided. Skipping evaluation.")
        return
    
    # Load ground truth
    df_gt = pd.read_csv(ground_truth_path)
    
    # Get predictions and ground truth
    predictions = [r['prediction'] for r in results if r['success']]
    ground_truth = df_gt['label'].tolist()[:len(predictions)]
    
    if len(predictions) != len(ground_truth):
        print(f"Warning: Mismatch in data length. Predictions: {len(predictions)}, Ground truth: {len(ground_truth)}")
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted')
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    print(f"\nConfusion Matrix:")
    print(cm)

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Bangla Fake News Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_path', type=str, help='Path to CSV data file')
    parser.add_argument('--text_column', type=str, default='headline', help='Text column name')
    parser.add_argument('--image_id_column', type=str, default='image_id', help='Image ID column name')
    parser.add_argument('--image_folder', type=str, default='images', help='Image folder path')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Output file path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--ground_truth', type=str, help='Path to ground truth file for evaluation')
    parser.add_argument('--single_text', type=str, help='Single text for prediction')
    parser.add_argument('--single_image', type=str, help='Single image path for prediction')
    
    args = parser.parse_args()
    
    # Initialize inference
    print("Initializing model...")
    batch_inference = BatchInference(args.model_path)
    realtime_inference = RealTimeInference(args.model_path)
    
    if args.single_text and args.single_image:
        # Single prediction
        print("Making single prediction...")
        result = realtime_inference.predict(args.single_text, args.single_image)
        
        if result['success']:
            print(f"\nPrediction Result:")
            print(f"Text: {args.single_text}")
            print(f"Image: {args.single_image}")
            print(f"Prediction: {result['label']} (Class: {result['prediction']})")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Inference Time: {result['inference_time']:.4f}s")
        else:
            print(f"Error: {result['error']}")
    
    elif args.data_path:
        # Batch prediction
        print("Loading data...")
        data = load_data_from_csv(
            args.data_path, 
            args.text_column, 
            args.image_id_column, 
            args.image_folder
        )
        
        print(f"Loaded {len(data)} samples")
        
        print("Making predictions...")
        start_time = time.time()
        results = batch_inference.predict_batch(data, args.batch_size)
        total_time = time.time() - start_time
        
        print(f"Completed {len(results)} predictions in {total_time:.2f}s")
        print(f"Average time per prediction: {total_time/len(results):.4f}s")
        
        # Save results
        save_results(results, args.output_path)
        
        # Evaluate if ground truth is provided
        if args.ground_truth:
            evaluate_model_performance(results, args.ground_truth)
        
        # Performance statistics
        perf_stats = realtime_inference.get_performance_stats()
        print(f"\nPerformance Statistics:")
        for key, value in perf_stats.items():
            print(f"{key}: {value}")
    
    else:
        print("Please provide either --single_text and --single_image for single prediction, or --data_path for batch prediction")

# Example usage functions
def example_single_prediction():
    """Example of single prediction"""
    inference = RealTimeInference('models/optimized_bangla_fake_news.pth')
    
    text = "এই একটি সত্য খবর"
    image_path = "images/sample.jpg"
    
    result = inference.predict(text, image_path)
    print(result)

def example_batch_prediction():
    """Example of batch prediction"""
    inference = BatchInference('models/optimized_bangla_fake_news.pth')
    
    data = [
        {'id': 1, 'text': 'খবর ১', 'image_path': 'images/image1.jpg'},
        {'id': 2, 'text': 'খবর ২', 'image_path': 'images/image2.jpg'}
    ]
    
    results = inference.predict_batch(data, batch_size=2)
    print(results)

if __name__ == "__main__":
    main()
