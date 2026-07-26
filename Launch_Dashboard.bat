@echo off
echo ==========================================
echo Syncing latest database from the cloud...
echo ==========================================
git pull origin main

echo.
echo ==========================================
echo Launching Quant Dashboard...
echo ==========================================
python -m streamlit run dashboard.py

pause