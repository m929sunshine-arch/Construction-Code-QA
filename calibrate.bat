@echo off
cd /d "%~dp0"
echo Building the index (first run about 1.5 hours) and calibrating refusal thresholds...
uv run python -m eval.calibrate --questions eval\questions_irc2018_full.jsonl
pause
