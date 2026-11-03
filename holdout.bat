@echo off
cd /d "%~dp0"
uv run python -m eval.run_eval --questions eval\questions_irc2018_holdout.jsonl --groups D
pause
