@echo off
echo Starting the Flask server...
pip install -r requirements.txt
flask --app mnscses run
pause