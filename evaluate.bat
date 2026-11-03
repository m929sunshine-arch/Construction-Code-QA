@echo off
cd /d "%~dp0"
uv run python -m eval.run_eval --questions eval\questions_irc2018_full.jsonl --groups B C D
pause
