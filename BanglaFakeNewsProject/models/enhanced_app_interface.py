# Enhanced Gradio Interface Definition
# This file contains the Gradio interface setup for the enhanced app

import gradio as gr
from enhanced_app import predict_fn, get_performance_dashboard, combined_css

# Create the enhanced Gradio interface
def create_enhanced_interface():
    """Create enhanced Gradio interface with monitoring dashboard"""
    
    with gr.Blocks(css=combined_css, title="Enhanced Bangla Fake News Detection") as demo:
        
        # Header
        with gr.Row():
            gr.HTML("""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 20px;">
                    <h1 style="color: white; font-size: 2.5rem; margin: 0; font-weight: 800;">
                        🔍 Enhanced Bangla Fake News Detection
                    </h1>
                    <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 8px 0 0 0;">
                        AI-powered news authenticity analysis with real-time monitoring
                    </p>
                </div>
            """)
        
        # Main content
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Enter News Content")
                gr.Markdown("Provide both headline and article text for best accuracy. Upload an image if available.")
                
                headline_input = gr.Textbox(
                    label="News Headline",
                    placeholder="Enter the news headline here... (e.g., ভূমিকম্পে কেঁপে উঠলো ঢাকা)",
                    lines=2
                )
                
                article_input = gr.Textbox(
                    label="News Article",
                    placeholder="Enter the full news article here... (minimum 3 words recommended)",
                    lines=4
                )
                
                image_input = gr.Image(
                    type="pil",
                    label="Upload News Image (Optional)",
                    height=200
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze News", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            # Right column - Results and monitoring
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Analysis Results")
                
                # Main result display
                result_output = gr.HTML(
                    value="<div style='text-align: center; padding: 40px; color: #64748b;'>Ready for analysis...</div>"
                )
                
                # Confidence gauge
                with gr.Row():
                    confidence_plot = gr.Plot(label="Confidence Gauge")
                
                # Real-time statistics
                with gr.Row():
                    stats_plot = gr.Plot(label="Prediction Statistics")
        
        # Performance Dashboard
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Performance Dashboard")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                
                performance_output = gr.Textbox(
                    label="Performance Report",
                    value=get_performance_dashboard(),
                    lines=10,
                    elem_classes=["performance-text"]
                )
        
        # Examples section
        gr.Markdown("---")
        gr.Markdown("### 💡 Example News Items")
        
        gr.Examples(
            examples=[
                [
                    "ভূমিকম্পে কেঁপে উঠলো ঢাকা",
                    "আজ সকালে ঢাকা শহরে এক শক্তিশালী ভূমিকম্প অনুভূত হয়েছে। রিখটার স্কেলে এর মাত্রা ছিল ৫.২। ভূমিকম্পের কারণে শহরের বিভিন্ন এলাকায় আতঙ্ক ছড়িয়ে পড়েছে।",
                    None
                ],
                [
                    "পদ্মা সেতুতে ফাটল দেখা দিয়েছে",
                    "সামাজিক যোগাযোগ মাধ্যমে পদ্মা সেতুতে ফাটলের গুজব ছড়িয়ে পড়েছে। তবে সেতু কর্তৃপক্ষ এই খবর সম্পূর্ণ অস্বীকার করেছে এবং বলেছে সেতু সম্পূর্ণ নিরাপদ।",
                    None
                ],
                [
                    "প্রধানমন্ত্রীর ভারত সফর",
                    "প্রধানমন্ত্রী শেখ হাসিনা আগামী সপ্তাহে ভারত সফরে যাবেন। এই সফরে দুই দেশের মধ্যে গুরুত্বপূর্ণ চুক্তি সই হওয়ার সম্ভাবনা রয়েছে।",
                    None
                ],
                [
                    "নতুন করোনা ভ্যারিয়েন্ট আবিষ্কার",
                    "বিজ্ঞানীরা করোনা ভাইরাসের একটি নতুন ভ্যারিয়েন্ট আবিষ্কার করেছেন যা আগের চেয়ে বেশি সংক্রামক। স্বাস্থ্য মন্ত্রণালয় সতর্কতা জারি করেছে।",
                    None
                ]
            ],
            inputs=[headline_input, article_input, image_input],
            outputs=[result_output, confidence_plot, stats_plot],
            fn=lambda h, a, i: predict_fn(f"{h} {a}", i),
            cache_examples=False
        )
        
        # Event handlers
        def combined_predict(headline, article, image):
            combined_text = f"{headline} {article}".strip()
            return predict_fn(combined_text, image)
        
        def clear_inputs():
            return "", "", None, "<div style='text-align: center; padding: 40px; color: #64748b;'>Ready for analysis...</div>", None, None
        
        def refresh_performance():
            return get_performance_dashboard()
        
        # Button events
        analyze_btn.click(
            fn=combined_predict,
            inputs=[headline_input, article_input, image_input],
            outputs=[result_output, confidence_plot, stats_plot]
        )
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[headline_input, article_input, image_input, result_output, confidence_plot, stats_plot]
        )
        
        refresh_btn.click(
            fn=refresh_performance,
            outputs=[performance_output]
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #e2e8f0; color: #64748b;">
                <p>Enhanced Bangla Fake News Detection System | Powered by Advanced AI Models</p>
                <p style="font-size: 0.9rem;">Real-time monitoring • Ensemble predictions • High accuracy</p>
            </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_enhanced_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )