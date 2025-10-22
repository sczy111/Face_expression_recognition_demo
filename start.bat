@echo off
REM ================================
REM Open Command Prompt inside virtual environment (relative paths)
REM ================================

REM Change directory to the folder where this batch file is located
cd /d "%~dp0"

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Keep the shell open for user commands
cmd
