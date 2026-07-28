@echo off
cd /d C:\Users\dando\Desktop\finetune_moirai

echo [1/4] Installing dependencies into finetune_moirai_env...
conda run -n finetune_moirai_env pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo [2/4] Downloading Moirai-large model locally (skipped if already exists)...
conda run -n finetune_moirai_env python download_model.py
if errorlevel 1 (
    echo ERROR: Model download failed.
    pause
    exit /b 1
)

echo [3/4] Preparing data...
conda run -n finetune_moirai_env python prepare_data.py
if errorlevel 1 (
    echo ERROR: Data preparation failed.
    pause
    exit /b 1
)

echo [4/4] Starting fine-tuning (close window to pause, progress is always saved)...
conda run -n finetune_moirai_env python finetune.py

pause
