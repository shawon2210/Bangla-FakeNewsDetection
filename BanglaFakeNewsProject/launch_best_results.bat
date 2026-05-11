@echo off
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║              🎯 BANGLA FAKE NEWS DETECTION - BEST RESULTS                   ║
echo ║                    Integrated Pipeline for Maximum Accuracy                  ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

echo 🔍 Validating pipeline integration...
python pipeline_integration.py

echo.
echo 🚀 Starting optimized system for best results...
echo.
echo 📊 System will be available at:
echo    Main App: http://localhost:7860
echo    Training Monitor: http://localhost:7861
echo.

python enhanced_launcher.py --full

echo.
echo 👋 Thank you for using the enhanced system!
pause