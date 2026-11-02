@echo off
REM Runs the quick suite: every test that needs no local data and is not slow,
REM the generated test map's test_map items included. This is what CI runs.
REM For the examples/ corpus and the slow checks too, see run_full_tests.bat.
REM See tests/README.md.
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
