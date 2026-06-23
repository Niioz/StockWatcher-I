@echo off
echo Starting The Council...
echo Checking dependencies...

set PYTHON_PATH="C:\Users\naron\AppData\Local\Programs\Python\Python312\python.exe"

if not exist venv (
    echo Creating virtual environment...
    %PYTHON_PATH% -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing required packages (this may take a minute on first run)...
pip install -r requirements.txt

echo Starting the Discord Secretary...
python bot.py
pause
