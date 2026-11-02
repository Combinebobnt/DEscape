@echo off
REM Alias kept for old instructions: runs run_quick_tests.bat with the same arguments.
echo run_all_tests is now run_quick_tests 1>&2
call "%~dp0run_quick_tests.bat" %*
exit /b %errorlevel%
