@echo off
REM Runs the default (fast, no-corpus) pytest tier. See tests/README.md for
REM what that covers. run_corpus_quick.bat adds the test map; for the
REM examples/ corpus, see run_corpus_stress.bat (~27 min).
REM
REM Parallel via pytest-xdist: one worker per CPU, capped at 8 so the machine
REM stays usable. Arguments come last, so -n 0 runs serially and
REM --maxprocesses 16 raises the cap.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv\Scripts\python.exe found - see LAUNCH_DEscape_Windows.bat or README.md to set one up. 1>&2
    exit /b 1
)

.venv\Scripts\python.exe -m pytest -n auto --maxprocesses 8 --dist loadgroup %*
