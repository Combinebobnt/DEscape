@echo off
REM Double-click-friendly launcher (Windows): finds Python 3 and hands off to
REM bootstrap.py, which creates the venv, installs dependencies, and starts
REM the editor (with a progress splash so a double-click launch isn't silent
REM while that happens) -- no manual `python -m venv` / `pip install` /
REM `.venv\Scripts\Activate.ps1` steps needed. See README.md.
REM
REM Assumes Python 3 itself is already installed (get it from
REM https://www.python.org/downloads/ if not, checking "Add python.exe to
REM PATH" during install) -- everything downstream of that is handled
REM automatically by bootstrap.py.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYTHON="
where py >nul 2>nul
if not errorlevel 1 set "PYTHON=py -3"
if not defined PYTHON (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON=python"
)

if not defined PYTHON (
    echo ===========================================================
    echo  Python 3 wasn't found on this system.
    echo  Install it from https://www.python.org/downloads/
    echo  During install, check "Add python.exe to PATH".
    echo  Then run this script again.
    echo ===========================================================
    pause
    exit /b 1
)

%PYTHON% bootstrap.py %*
if errorlevel 1 pause

endlocal
