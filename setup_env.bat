@echo off

echo ================================
echo Setting up Trading Bot Environment
echo ================================

python -m venv .venv

call .venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

echo.
echo Setup complete.
echo To activate manually in future:
echo .venv\Scripts\activate
pause