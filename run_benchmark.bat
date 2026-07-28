@echo off
cd /d C:\Users\dando\Desktop\finetune_moirai

echo Installing minimum deps for benchmark...
conda run -n finetune_moirai_env pip install torch pyyaml pyarrow numpy --quiet

echo.
echo Running benchmark...
echo.
conda run -n finetune_moirai_env python benchmark.py

pause
