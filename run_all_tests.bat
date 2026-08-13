@echo off
REM Runs the default (fast, no-corpus) pytest tier. See tests/README.md for
REM what that covers. For the corpus/gui tier, see run_corpus_quick.bat
REM (~15 min, the default scope) and run_corpus_stress.bat (~27 min, the
REM complete corpus).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv\Scripts\python.exe found - see LAUNCH_DESCAPE_Windows.bat or README.md to set one up. 1>&2
    exit /b 1
)

.venv\Scripts\python.exe -m pytest %*
