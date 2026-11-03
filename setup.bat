@echo off
cd /d "%~dp0"
echo [1/4] Creating virtual environment...
uv venv
if errorlevel 1 goto fail
echo [2/4] Installing packages (can take 10+ minutes)...
uv pip install -r requirements.txt -r requirements-models.txt
if errorlevel 1 goto fail
if exist .env goto haveenv
echo [3/4] Creating .env
set /p KEY=Paste your DeepSeek API key, then press Enter: 
> .env echo DEEPSEEK_API_KEY=%KEY%
>> .env echo HF_HUB_OFFLINE=1
>> .env echo IRC_RAG_CONFIG=config.irc2018.yaml
:haveenv
echo [4/4] Running tests...
uv run pytest -q
if errorlevel 1 goto fail
echo.
echo Setup finished. Next: double-click calibrate.bat (first run builds the index, about 1.5 hours).
pause
exit /b 0
:fail
echo.
echo Something went wrong. Take a screenshot of this window.
pause
exit /b 1
