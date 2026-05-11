@echo off
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    🚀 ENHANCED BANGLA FAKE NEWS DETECTION                    ║
echo ║                           Quick Launch Menu                                  ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

:menu
echo Please select an option:
echo.
echo 1. 🚀 Start Enhanced App (Recommended)
echo 2. 🏋️  Start Enhanced Training
echo 3. 📊 Start Training Monitor
echo 4. 🎯 Start Full System (App + Monitor)
echo 5. 🧪 Run System Test
echo 6. 🔍 Check Requirements
echo 7. ❌ Exit
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto start_app
if "%choice%"=="2" goto start_training
if "%choice%"=="3" goto start_monitor
if "%choice%"=="4" goto start_full
if "%choice%"=="5" goto run_test
if "%choice%"=="6" goto check_req
if "%choice%"=="7" goto exit
echo Invalid choice. Please try again.
goto menu

:start_app
echo.
echo 🚀 Starting Enhanced App...
python enhanced_launcher.py --app
goto end

:start_training
echo.
echo 🏋️ Starting Enhanced Training...
echo ⚠️  This will take several hours. Training monitor will be available at http://localhost:7861
python enhanced_launcher.py --train
goto end

:start_monitor
echo.
echo 📊 Starting Training Monitor...
echo Monitor will be available at http://localhost:7861
python enhanced_launcher.py --monitor
goto end

:start_full
echo.
echo 🎯 Starting Full System...
echo App: http://localhost:7860
echo Monitor: http://localhost:7861
python enhanced_launcher.py --full
goto end

:run_test
echo.
echo 🧪 Running System Test...
python enhanced_launcher.py --test
pause
goto menu

:check_req
echo.
echo 🔍 Checking Requirements...
python enhanced_launcher.py --check
pause
goto menu

:exit
echo.
echo 👋 Thank you for using Enhanced Bangla Fake News Detection!
goto end

:end
echo.
echo Press any key to exit...
pause >nul