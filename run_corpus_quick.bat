@echo off
REM Runs the corpus/gui pytest tier against the default QUICK_CORPUS_NAMES
REM subset (~15 min) -- see tests/README.md and tests/conftest.py. For the
REM complete corpus (~27 min), use run_corpus_stress.bat instead.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv\Scripts\python.exe found - see LAUNCH_DESCAPE_Windows.bat or README.md to set one up. 1>&2
    exit /b 1
)

.venv\Scripts\python.exe -m pytest -m "corpus or slow" %*
