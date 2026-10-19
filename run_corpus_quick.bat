@echo off
REM Runs the default tier plus the generated test map's test_map items and
REM the slow checks, in parallel like run_all_tests.bat. No examples/ map runs
REM here: the real corpus is run_corpus_stress.bat's (~27 min). See
REM tests/README.md and tests/conftest.py.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv\Scripts\python.exe found - see LAUNCH_DEscape_Windows.bat or README.md to set one up. 1>&2
    exit /b 1
)

.venv\Scripts\python.exe -m pytest -n auto --maxprocesses 8 --dist loadgroup -m "not corpus" %*
