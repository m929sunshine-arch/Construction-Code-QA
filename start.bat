@echo off
cd /d "%~dp0"
uv run streamlit run app\streamlit_app.py
pause
