@echo off
echo 🧠 Setting up SignBridge environment...

:: Check if running in Colab
python -c "import sys; print('google.colab' in sys.modules)" > temp.txt
set /p IS_COLAB=<temp.txt
del temp.txt

:: Create virtual environment (if not in Colab)
if not "%IS_COLAB%"=="True" (
    if not exist venv (
        echo Creating virtual environment...
        python -m venv venv
        call venv\Scripts\activate
        echo Virtual environment created and activated.
    ) else (
        echo Virtual environment already exists.
        call venv\Scripts\activate
    )
)

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Install Colab-specific dependencies if in Colab
if "%IS_COLAB%"=="True" (
    echo Installing Colab-specific dependencies...
    pip install pyngrok
)

:: Install MediaPipe with Holistic model
pip install mediapipe

:: Mount Google Drive if in Colab
if "%IS_COLAB%"=="True" (
    echo Mounting Google Drive...
    python -c "from google.colab import drive; drive.mount('/content/drive')"
    
    :: Create base directory structure if it doesn't exist
    set BASE_DIR=/content/drive/MyDrive/SignBridge_Data
    
    python -c "import os; base_dir='%BASE_DIR%'; os.makedirs(f'{base_dir}/metadata', exist_ok=True); os.makedirs(f'{base_dir}/sign_landmarks/train', exist_ok=True); os.makedirs(f'{base_dir}/sign_landmarks/val', exist_ok=True); os.makedirs(f'{base_dir}/sign_landmarks/test', exist_ok=True); os.makedirs(f'{base_dir}/speech/raw_audio', exist_ok=True); os.makedirs(f'{base_dir}/speech/transcripts', exist_ok=True); os.makedirs(f'{base_dir}/models/sign', exist_ok=True); os.makedirs(f'{base_dir}/models/speech', exist_ok=True); print('Directory structure created successfully!')"
    
    :: Create a symbolic link for easy access
    mklink /D SignBridge_Data "%BASE_DIR%"
) else (
    echo Not running in Colab. Skipping Google Drive mount.
)

echo ✅ SignBridge environment setup complete!