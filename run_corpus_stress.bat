@echo off
REM Runs the corpus/gui pytest tier against the COMPLETE examples/ corpus
REM (~27 min), plus the generated test map's test_map items -- see
REM tests/README.md and tests/conftest.py. For routine after-a-feature
REM verification, use run_corpus_quick.bat instead (default tier plus the
REM test map, no examples/); reach for this before a release or after
REM touching rendering/write-path internals.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv\Scripts\python.exe found - see LAUNCH_DEscape_Windows.bat or README.md to set one up. 1>&2
    exit /b 1
)

.venv\Scripts\python.exe -m pytest -m "corpus or slow or test_map" --corpus-full %*
