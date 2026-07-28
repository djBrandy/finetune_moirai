@echo off
cd /d C:\Users\dando\Desktop\finetune_moirai

echo Activating finetune_moirai_env...
call conda activate finetune_moirai_env

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo Preparing data...
python prepare_data.py

echo Starting fine-tuning (Ctrl+C to pause, progress is saved)...
python finetune.py

pause
