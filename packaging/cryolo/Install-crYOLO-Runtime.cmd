@echo off
setlocal
rem Repair helper only; the main installer and crYOLO profile prepare this automatically.
echo Preparing the private crYOLO runtime automatically...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-runtime.ps1" %*
set "SARU_RUNTIME_EXIT=%ERRORLEVEL%"
if not "%SARU_RUNTIME_EXIT%"=="0" (
  echo crYOLO setup failed. See runtime-install-report.json in the SARU runtime folder.
) else (
  echo crYOLO is ready. Open SARU Picker and select the crYOLO profile.
)
exit /b %SARU_RUNTIME_EXIT%
