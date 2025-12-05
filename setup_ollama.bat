@echo off
REM Ollama Setup Script for Windows
echo ========================================
echo Ollama Setup for Patient Discharge Assistant
echo ========================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ollama is not installed or not in PATH
    echo.
    echo Please install Ollama from: https://ollama.ai
    echo After installation, restart this script.
    pause
    exit /b 1
)

echo [OK] Ollama is installed
echo.

REM Check Ollama version
echo Checking Ollama version...
ollama --version
echo.

REM Check if Ollama server is running
echo Checking if Ollama server is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ollama server is not running
    echo.
    echo Starting Ollama server...
    echo NOTE: Keep this window open, or start Ollama manually with: ollama serve
    start /B ollama serve
    timeout /t 3 /nobreak >nul
    echo.
)

REM Check again
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not connect to Ollama server
    echo.
    echo Please start Ollama manually:
    echo   1. Open a new terminal
    echo   2. Run: ollama serve
    echo   3. Keep that terminal open
    echo   4. Then run this script again
    pause
    exit /b 1
)

echo [OK] Ollama server is running
echo.

REM List installed models
echo Checking installed models...
ollama list
echo.

REM Check if llama3.2 is installed
ollama list | findstr /C:"llama3.2" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] llama3.2 model is not installed
    echo.
    set /p install="Do you want to install llama3.2 now? (y/n): "
    if /i "%install%"=="y" (
        echo.
        echo Downloading llama3.2 model (this may take several minutes)...
        ollama pull llama3.2
        if %ERRORLEVEL% EQU 0 (
            echo [OK] llama3.2 installed successfully
        ) else (
            echo [ERROR] Failed to install llama3.2
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo You can install it later with: ollama pull llama3.2
    )
) else (
    echo [OK] llama3.2 model is installed
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run: python check_setup.py
echo   2. Start the server: python start.py
echo   3. Open browser: http://localhost:8000
echo.
pause


