@echo off
setlocal
echo Installing the separate crYOLO CPU runtime...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-runtime.ps1"
set "SARU_RUNTIME_EXIT=%ERRORLEVEL%"
if not "%SARU_RUNTIME_EXIT%"=="0" (
  echo.
  echo Runtime installation failed. See the message above and runtime-install-report.json.
) else (
  echo.
  echo Runtime installation finished. You can launch SARU Picker crYOLO.
)
pause
exit /b %SARU_RUNTIME_EXIT%
