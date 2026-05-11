import gradio as gr
import tempfile
import os
import sys
from pathlib import Path

# Resolve absolute path to config.yaml relative to this file
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = str(BASE_DIR / "config.yaml")

# Add models directory to sys.path for imports
MODELS_DIR = str(BASE_DIR / "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

# Use the existing predictor
from predictor import Predictor

# Try different config paths
config_paths = [CONFIG_PATH, "config.yaml", "../config.yaml"]
config_found = None
for path in config_paths:
    if os.path.exists(path):
        config_found = path
        break

if not config_found:
    raise FileNotFoundError("config.yaml not found")

predictor = Predictor(config_path=config_found)
print("✅ Using predictor")

# Define the prediction function
def predict_fn(text_combined, image):
    text = text_combined.strip()
    image_path = None
    user_tips = []

    # Input validation and user guidance
    if not text or len(text.split()) < 5:
        user_tips.append("Please enter a detailed news headline and article (at least 5 words) for best results.")
    if image is None:
        user_tips.append("Uploading a relevant news image can improve prediction accuracy.")

    try:
        # Handle image - only if it's a PIL Image object
        if image is not None and hasattr(image, 'save'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                image.save(tmpfile.name)
                image_path = tmpfile.name
        # Use existing prediction strategy with enhancements
        result = predictor.predict(text, image_path)
        
        # Enhanced result processing while maintaining compatibility
        if 'model_count' in result and result['model_count'] > 1:
            user_tips.append(f"🤖 {result['model_count']} AI models analyzed this news for better accuracy")
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"<div class='result-card'><div class='result-label'>Error: {str(e)}</div></div>"
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

    prediction = result['prediction']
    confidence = result['confidence']
    news_type = result.get('news_type', 'General')
    model_count = result.get('model_count', 1)
    
    # Enhanced confidence interpretation
    if confidence >= 0.9:
        confidence_level = "Very High"
    elif confidence >= 0.8:
        confidence_level = "High"
    elif confidence >= 0.7:
        confidence_level = "Good"
    elif confidence >= 0.6:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    if prediction == 'Real':
        prediction_class = "prediction-real"
        icon = "✅"
    else:
        prediction_class = "prediction-fake"
        icon = "❌"

    # Enhanced actionable feedback
    if confidence < 0.6:
        user_tips.append("⚠️ Low confidence detected. Consider providing more detailed text or a relevant image.")
    elif confidence < 0.8:
        user_tips.append("📊 Moderate confidence. The analysis is reasonably reliable.")
    else:
        user_tips.append("🎯 High confidence result. The analysis is very reliable.")

    # Modern result card with icon, news type, and confidence
    result_card = f"""
    <div class='result-card {prediction_class}'>
        <div class='result-icon'>{icon}</div>
        <div class='result-type'>{news_type}</div>
        <div class='result-label'>{prediction}</div>
        <div class='confidence-gauge-container'>
            <div class='confidence-gauge' style='width: {confidence * 100}%;'></div>
        </div>
        <div class='result-confidence'>Confidence: {confidence:.2%} ({confidence_level})</div>
        <div class='model-info'>Models: {model_count} | Enhanced Analysis</div>
        {('<ul class="user-tips">' + ''.join(f'<li>{tip}</li>' for tip in user_tips) + '</ul>') if user_tips else ''}
    </div>
    """
    return result_card
# Ensure CSS is loaded relative to this file (works even when cwd differs)
css_path = Path(__file__).resolve().parent / "style.css"
css_arg = str(css_path) if css_path.exists() else "style.css"

# Define the Gradio interface with Blocks
with gr.Blocks(css=css_arg) as demo:


    with gr.Column(elem_classes=["app-shell"]):
        # Header
        with gr.Row(elem_classes=["header"]):
            with gr.Column(elem_classes=["brand"]):
                gr.HTML("""
                    <div class="brand-logo">BN</div>
                    <div>
                        <div class="brand-title">Bangla Fake News Detection</div>
                        <div class="brand-sub">AI-powered news authenticity analysis</div>
                    </div>
                """)

        # Main content panel
        with gr.Column(elem_classes=["panel"]):
            with gr.Row(elem_classes=["grid"]):
                # Input column
                with gr.Column():
                    gr.Markdown("### Enter News Details")
                    gr.Markdown("For best accuracy, provide both a detailed headline and the full news article. Upload a relevant image if available.")
                    text_input = gr.Textbox(label="News Headline", placeholder="Enter the news headline here... (e.g., ভূমিকম্পে কেঁপে উঠলো ঢাকা)")
                    article_input = gr.Textbox(label="News Article", placeholder="Enter the full news article here... (at least 5 words)")
                    image_input = gr.Image(
                        type="pil", 
                        label="Upload News Image (Optional, but recommended)",
                        elem_classes=["card"]
                    )
                    with gr.Row(elem_classes=["footer"]):
                        submit_button = gr.Button("Analyze News", variant="primary")
                        clear_button = gr.Button("Clear", elem_classes=["btn-ghost"])

                # Prediction column
                with gr.Column():
                    gr.Markdown("### Analysis Result")
                    with gr.Column(elem_classes=["prediction"]):
                        prediction_output = gr.HTML(label="Result", elem_id="result-card", value="<div class='placeholder'>Awaiting analysis...</div>")

    gr.Markdown("---")
    gr.Markdown("Examples")
    gr.Examples(
        examples=[
            ["ভূমিকম্পে কেঁপে উঠলো ঢাকা", "আজ সকালে ঢাকা শহরে এক শক্তিশালী ভূমিকম্প অনুভূত হয়েছে। রিখটার স্কেলে এর মাত্রা ছিল ৫.২।", None],
            ["পদ্মা সেতুতে ফাটল", "সামাজিক যোগাযোগ মাধ্যমে পদ্মা সেতুতে ফাটলের গুজব ছড়িয়ে পড়েছে। সেতু কর্তৃপক্ষ এই খবর অস্বীকার করেছে।", None],
            ["প্রধানমন্ত্রীর ভারত সফরে ৪ চুক্তি সই", "", None],
            ["বিআরটিসির বহরে যুক্ত হচ্ছে নতুন বাস", "", None],
            ["আওয়ামী লীগের নতুন কমিটিতে চমক", "", None],
        ],
        inputs=[text_input, article_input, image_input],
        outputs=[prediction_output],
        fn=lambda headline, article, image: predict_fn(f"{headline} {article}", image),
        cache_examples=False,
    )

    # Define the button actions
    submit_button.click(fn=lambda headline, article, image: predict_fn(f"{headline} {article}", image), inputs=[text_input, article_input, image_input], outputs=[prediction_output])
    clear_button.click(lambda: [None, None, None, None], None, [text_input, article_input, image_input, prediction_output])

if __name__ == "__main__":
    # Clear any cached examples to avoid 403 errors
    import shutil
    cache_dir = Path(__file__).parent / "gradio_cached_examples"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    
    # Try multiple ports in case default is busy
    ports_to_try = [7860, 7862, 7863, 7864, 7865]
    
    for port in ports_to_try:
        try:
            print(f"🌐 Launching Gradio app on http://localhost:{port}")
            demo.launch(server_port=port)
            break  # Success
        except Exception as e:
            if "Cannot find empty port" in str(e) and port != ports_to_try[-1]:
                continue
            else:
                print(f"Failed to launch on port {port}: {e}")
                break
