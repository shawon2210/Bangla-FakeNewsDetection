"""
Comprehensive Model Evaluation and Visualization Tools
For Bangla Fake News Detection Research
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation with visualization"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['Real', 'Fake']
        self.results = {}
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: np.ndarray = None) -> Dict:
        """Comprehensive evaluation of model predictions"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # ROC and PR curves if probabilities provided
        if y_prob is not None:
            roc_results = self._compute_roc_metrics(y_true, y_prob)
            pr_results = self._compute_pr_metrics(y_true, y_prob)
            results.update(roc_results)
            results.update(pr_results)
        
        self.results = results
        return results
    
    def _compute_roc_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Compute ROC curve metrics"""
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                
                return {
                    'roc_auc': roc_auc,
                    'roc_fpr': fpr.tolist(),
                    'roc_tpr': tpr.tolist()
                }
            else:
                # Multi-class ROC
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                return {'roc_auc': roc_auc}
        except Exception as e:
            print(f"Error computing ROC metrics: {e}")
            return {}
    
    def _compute_pr_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Compute Precision-Recall curve metrics"""
        try:
            if len(np.unique(y_true)) == 2:
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                avg_precision = average_precision_score(y_true, y_prob[:, 1])
                
                return {
                    'average_precision': avg_precision,
                    'pr_precision': precision.tolist(),
                    'pr_recall': recall.tolist()
                }
            else:
                avg_precision = average_precision_score(y_true, y_prob, average='weighted')
                return {'average_precision': avg_precision}
        except Exception as e:
            print(f"Error computing PR metrics: {e}")
            return {}
    
    def plot_confusion_matrix(self, save_path: str = None, figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix"""
        if 'confusion_matrix' not in self.results:
            raise ValueError("No confusion matrix found. Run evaluate_predictions first.")
        
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=figsize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, save_path: str = None, figsize: Tuple[int, int] = (8, 6)):
        """Plot ROC curve"""
        if 'roc_fpr' not in self.results:
            print("No ROC curve data available.")
            return
        
        plt.figure(figsize=figsize)
        fpr = self.results['roc_fpr']
        tpr = self.results['roc_tpr']
        roc_auc = self.results['roc_auc']
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: str = None, figsize: Tuple[int, int] = (8, 6)):
        """Plot Precision-Recall curve"""
        if 'pr_precision' not in self.results:
            print("No PR curve data available.")
            return
        
        plt.figure(figsize=figsize)
        precision = self.results['pr_precision']
        recall = self.results['pr_recall']
        avg_precision = self.results['average_precision']
        
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_metrics(self, save_path: str = None, figsize: Tuple[int, int] = (12, 8)):
        """Plot per-class metrics"""
        if 'precision_per_class' not in self.results:
            raise ValueError("No per-class metrics found. Run evaluate_predictions first.")
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        precision = self.results['precision_per_class']
        recall = self.results['recall_per_class']
        f1 = self.results['f1_per_class']
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: List[float],
                               save_path: str = None, figsize: Tuple[int, int] = (10, 6)):
        """Plot feature importance"""
        plt.figure(figsize=figsize)
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_scores = [importance_scores[i] for i in sorted_indices]
        
        bars = plt.barh(range(len(sorted_names)), sorted_scores)
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_predictions first.")
        
        report = []
        report.append("=" * 60)
        report.append("BANGLA FAKE NEWS DETECTION - MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 30)
        report.append(f"Accuracy: {self.results['accuracy']:.4f}")
        report.append(f"Precision (Weighted): {self.results['precision_weighted']:.4f}")
        report.append(f"Recall (Weighted): {self.results['recall_weighted']:.4f}")
        report.append(f"F1-Score (Weighted): {self.results['f1_weighted']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS PERFORMANCE:")
        report.append("-" * 30)
        for i, class_name in enumerate(self.class_names):
            precision = self.results['precision_per_class'][i]
            recall = self.results['recall_per_class'][i]
            f1 = self.results['f1_per_class'][i]
            support = self.results['support_per_class'][i]
            
            report.append(f"{class_name}:")
            report.append(f"  Precision: {precision:.4f}")
            report.append(f"  Recall: {recall:.4f}")
            report.append(f"  F1-Score: {f1:.4f}")
            report.append(f"  Support: {support}")
            report.append("")
        
        # ROC and PR metrics
        if 'roc_auc' in self.results:
            report.append("ROC CURVE METRICS:")
            report.append("-" * 30)
            report.append(f"ROC AUC: {self.results['roc_auc']:.4f}")
            report.append("")
        
        if 'average_precision' in self.results:
            report.append("PRECISION-RECALL METRICS:")
            report.append("-" * 30)
            report.append(f"Average Precision: {self.results['average_precision']:.4f}")
            report.append("")
        
        # Confusion matrix
        report.append("CONFUSION MATRIX:")
        report.append("-" * 30)
        cm = np.array(self.results['confusion_matrix'])
        report.append("Predicted ->")
        report.append("Actual    Real  Fake")
        for i, class_name in enumerate(self.class_names):
            row_str = f"{class_name:8} {cm[i, 0]:4d} {cm[i, 1]:4d}"
            report.append(row_str)
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Evaluation report saved to: {save_path}")
        
        return report_text
    
    def save_results(self, save_path: str):
        """Save evaluation results to JSON"""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Evaluation results saved to: {save_path}")

def compare_models(results_dict: Dict[str, Dict], save_path: str = None, 
                  figsize: Tuple[int, int] = (15, 10)):
    """Compare multiple model results"""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model].get(metric, 0) for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(models, key=len)) > 10:
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_prediction_errors(y_true: np.ndarray, y_pred: np.ndarray, 
                            texts: List[str], save_path: str = None):
    """Analyze prediction errors for insights"""
    errors = []
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            errors.append({
                'index': i,
                'true_label': true,
                'predicted_label': pred,
                'text': texts[i] if i < len(texts) else f"Sample {i}"
            })
    
    error_df = pd.DataFrame(errors)
    
    if len(error_df) > 0:
        print(f"Total prediction errors: {len(error_df)}")
        print("\nError Analysis:")
        print(error_df.groupby(['true_label', 'predicted_label']).size())
        
        if save_path:
            error_df.to_csv(save_path, index=False)
            print(f"Error analysis saved to: {save_path}")
    else:
        print("No prediction errors found!")

# Example usage
def example_evaluation():
    """Example of how to use the evaluator"""
    # Sample data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6],
                      [0.3, 0.7], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]])
    
    # Initialize evaluator
    evaluator = ModelEvaluator(['Real', 'Fake'])
    
    # Evaluate
    results = evaluator.evaluate_predictions(y_true, y_pred, y_prob)
    
    # Generate plots
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()
    evaluator.plot_precision_recall_curve()
    evaluator.plot_class_metrics()
    
    # Generate report
    report = evaluator.generate_report('evaluation_report.txt')
    print(report)
    
    # Save results
    evaluator.save_results('evaluation_results.json')

if __name__ == "__main__":
    example_evaluation()
