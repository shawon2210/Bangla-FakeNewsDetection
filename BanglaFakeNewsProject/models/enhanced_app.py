"""
Enhanced Gradio App with Real-time Performance Monitoring and Better UI
Improved frontend visibility and user experience
"""

import gradio as gr
import tempfile
import os
import json
import time
from pathlib import Path
import plotly.graph_objs as go
import plotly.express as px
from enhanced_predictor import EnhancedPredictor

# Resolve absolute path to config.yaml relative to this file (parent of models directory)
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(BASE_DIR / "config.yaml")

# Initialize enhanced predictor
try:
    predictor = EnhancedPredictor(config_path=CONFIG_PATH)
    print("✅ Enhanced predictor loaded successfully")
except Exception as e:
    print(f"❌ Failed to load enhanced predictor: {e}")
    # Fallback to original predictor
    from predictor import Predictor
    predictor = Predictor(config_path=CONFIG_PATH)
    print("⚠️ Using fallback predictor")

# Global variables for real-time monitoring
prediction_stats = {
    'total_predictions': 0,
    'real_count': 0,
    'fake_count': 0,
    'avg_confidence': 0.0,
    'recent_predictions': []
}

def update_stats(result):
    """Update global statistics"""
    global prediction_stats
    
    prediction_stats['total_predictions'] += 1
    
    if result['prediction'] == 'Real':
        prediction_stats['real_count'] += 1
    else:
        prediction_stats['fake_count'] += 1
    
    # Update average confidence
    total = prediction_stats['total_predictions']
    current_avg = prediction_stats['avg_confidence']
    prediction_stats['avg_confidence'] = (current_avg * (total - 1) + result['confidence']) / total
    
    # Store recent predictions (last 10)
    prediction_stats['recent_predictions'].append({
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'news_type': result.get('news_type', 'General'),
        'timestamp': time.time()
    })
    
    if len(prediction_stats['recent_predictions']) > 10:
        prediction_stats['recent_predictions'] = prediction_stats['recent_predictions'][-10:]

def create_confidence_gauge(confidence):
    """Create confidence gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_stats_chart():
    """Create real-time statistics chart"""
    if prediction_stats['total_predictions'] == 0:
        return None
    
    # Pie chart for Real vs Fake
    labels = ['Real', 'Fake']
    values = [prediction_stats['real_count'], prediction_stats['fake_count']]
    colors = ['#16a34a', '#dc2626']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="Prediction Distribution",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def predict_fn(text_combined, image):
    """Enhanced prediction function with monitoring"""
    text = text_combined.strip()
    image_path = None
    user_tips = []
    
    # Input validation
    if not text or len(text.split()) < 3:
        user_tips.append("⚠️ Please enter at least 3 words for better accuracy")
    
    if image is None:
        user_tips.append("💡 Adding an image can improve prediction accuracy")
    
    try:
        # Handle image
        if image is not None and hasattr(image, 'save'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                image.save(tmpfile.name)
                image_path = tmpfile.name
        
        # Make prediction
        start_time = time.time()
        result = predictor.predict(text, image_path)
        inference_time = time.time() - start_time
        
        # Update statistics
        update_stats(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return create_error_card(str(e)), None, None
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    
    # Extract results
    prediction = result['prediction']
    confidence = result['confidence']
    news_type = result.get('news_type', 'General')
    model_count = result.get('model_count', 1)
    
    # Determine prediction class and icon
    if prediction == 'Real':
        prediction_class = "prediction-real"
        icon = "✅"
        color = "#16a34a"
    else:
        prediction_class = "prediction-fake"
        icon = "❌"
        color = "#dc2626"
    
    # Add confidence-based tips
    if confidence < 0.6:
        user_tips.append("🤔 Low confidence - consider providing more context")
    elif confidence < 0.8:
        user_tips.append("📊 Moderate confidence - result is reasonably reliable")
    else:
        user_tips.append("🎯 High confidence - result is highly reliable")
    
    # Enhanced result card with more information
    result_card = f"""
    <div class='enhanced-result-card {prediction_class}'>
        <div class='result-header'>
            <div class='result-icon'>{icon}</div>
            <div class='result-info'>
                <div class='result-label'>{prediction}</div>
                <div class='result-type'>{news_type} News</div>
            </div>
        </div>
        
        <div class='confidence-section'>
            <div class='confidence-label'>Confidence Score</div>
            <div class='confidence-bar-container'>
                <div class='confidence-bar' style='width: {confidence * 100}%; background-color: {color};'></div>
            </div>
            <div class='confidence-text'>{confidence:.1%}</div>
        </div>
        
        <div class='model-info'>
            <div class='model-count'>🤖 {model_count} Model{'s' if model_count > 1 else ''} Used</div>
            <div class='inference-time'>⚡ {inference_time:.3f}s</div>
        </div>
        
        {('<div class="user-tips">' + ''.join(f'<div class="tip">{tip}</div>' for tip in user_tips) + '</div>') if user_tips else ''}
    </div>
    """
    
    # Create visualizations
    confidence_gauge = create_confidence_gauge(confidence)
    stats_chart = create_stats_chart()
    
    return result_card, confidence_gauge, stats_chart

def get_performance_dashboard():
    """Get performance dashboard data"""
    if hasattr(predictor, 'get_performance_report'):
        return predictor.get_performance_report()
    else:
        return f"""
📊 Basic Performance Stats
========================
Total Predictions: {prediction_stats['total_predictions']}
Real News: {prediction_stats['real_count']}
Fake News: {prediction_stats['fake_count']}
Average Confidence: {prediction_stats['avg_confidence']:.3f}
        """

def create_error_card(error_msg):
    """Create error display card"""
    return f"""
    <div class='error-card'>
        <div class='error-icon'>⚠️</div>
        <div class='error-message'>
            <div class='error-title'>Prediction Error</div>
            <div class='error-details'>{error_msg}</div>
        </div>
    </div>
    """

# Enhanced CSS
enhanced_css = """
.enhanced-result-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 16px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    padding: 24px;
    margin: 16px 0;
    border: 2px solid #e2e8f0;
    transition: all 0.3s ease;
    font-family: 'Inter', system-ui, sans-serif;
}

.enhanced-result-card.prediction-real {
    border-color: #16a34a;
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
}

.enhanced-result-card.prediction-fake {
    border-color: #dc2626;
    background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
}

.result-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
}

.result-icon {
    font-size: 3rem;
    line-height: 1;
}

.result-info {
    flex: 1;
}

.result-label {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 4px;
}

.result-type {
    font-size: 1.1rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.confidence-section {
    margin-bottom: 20px;
}

.confidence-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.confidence-bar-container {
    background: #e2e8f0;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin-bottom: 8px;
}

.confidence-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

.confidence-text {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e293b;
    text-align: center;
}

.model-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: rgba(0,0,0,0.02);
    border-radius: 8px;
    margin-bottom: 16px;
}

.model-count, .inference-time {
    font-size: 0.9rem;
    font-weight: 600;
    color: #475569;
}

.user-tips {
    border-top: 1px solid #e2e8f0;
    padding-top: 16px;
}

.tip {
    background: #f1f5f9;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.9rem;
    color: #475569;
    border-left: 3px solid #3b82f6;
}

.error-card {
    display: flex;
    align-items: center;
    gap: 16px;
    background: #fef2f2;
    border: 2px solid #fecaca;
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
}

.error-icon {
    font-size: 2rem;
    color: #dc2626;
}

.error-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #dc2626;
    margin-bottom: 4px;
}

.error-details {
    font-size: 0.9rem;
    color: #7f1d1d;
}

.stats-container {
    background: #f8fafc;
    border-radius: 12px;
    padding: 16px;
    margin-top: 16px;
}

.performance-text {
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    line-height: 1.4;
    background: #1e293b;
    color: #e2e8f0;
    padding: 16px;
    border-radius: 8px;
    white-space: pre-wrap;
}
"""

# Load existing CSS and combine
css_path = Path(__file__).resolve().parent / "style.css"
existing_css = ""
if css_path.exists():
    with open(css_path, 'r') as f:
        existing_css = f.read()

combined_css = existing_css + "\n" + enhanced_css