@echo off
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    🔄 RESUME TRAINING - BANGLA FAKE NEWS                     ║
echo ║                        Continue from where you left off                      ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

echo Please select training approach:
echo.
echo 1. 🚀 Resume Enhanced Training (Recommended - Better accuracy)
echo 2. 🔄 Resume Original Training (Your existing approach)
echo 3. 📊 Start Training Monitor Only
echo 4. ❌ Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto enhanced_training
if "%choice%"=="2" goto original_training
if "%choice%"=="3" goto monitor_only
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
goto menu

:enhanced_training
echo.
echo 🚀 Starting Enhanced Training (with real-time monitoring)...
echo 📊 Monitor will be available at: http://localhost:7861
echo.
python resume_enhanced_training.py
goto end

:original_training
echo.
echo 🔄 Resuming Original Training Approach...
echo.
python resume_original_training.py
goto end

:monitor_only
echo.
echo 📊 Starting Training Monitor Only...
echo Monitor available at: http://localhost:7861
echo.
python training_monitor.py
goto end

:exit
echo.
echo 👋 Goodbye!
goto end

:end
echo.
echo Press any key to exit...
pause >nul