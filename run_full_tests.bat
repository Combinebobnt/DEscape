@echo off
REM Runs the full suite: the quick suite plus the slow checks and the
REM COMPLETE examples/ corpus, in one parallel run. The run before a release or
REM after touching rendering or write-path internals; see tests/README.md for
REM its time and tests/conftest.py for the corpus.
REM
REM Parallel like run_quick_tests.bat, capped at 8 workers. Arguments come
REM last, so -n 0 runs serially, --maxprocesses 4 suits a machine with less
REM RAM, -m corpus narrows it and --scenario-dir= points it at other files.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv\Scripts\python.exe found - see LAUNCH_DEscape_Windows.bat or README.md to set one up. 1>&2
    exit /b 1
)

REM A tautology rather than -m "": selects every item.
.venv\Scripts\python.exe -m pytest -n auto --maxprocesses 8 --dist loadgroup -m "corpus or not corpus" --corpus-full %*
