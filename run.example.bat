@echo off
setlocal

:: Copy this file to run.bat and fill in the paths for your machine.

:: Path to your @runanywhere/electron dist/ build.
set "RCLI_XL8_SDK_DIST=C:/path/to/runanywhere-electron/dist"

:: RunAnywhere LLM catalog id or local GGUF path.
set "RCLI_XL8_LLM_PATH=qwen2.5-3b"

:: Optional: real python.exe if `python` is the Windows Store stub.
set "RCLI_XL8_PYTHON=python"

:: Target language for live translations (default en).
set "RCLI_XL8_TO=en"

:: Label for the other person in captions.
set "RCLI_XL8_OTHER=other"

:: Optional: reuse Whisper assets from an existing rcli-meet install.
:: set "RCLI_XL8_WHISPER_BIN=D:/the_code/runanywhere/SDK/rcli-meet/bin/whisper"
:: set "RCLI_XL8_WHISPER_MODEL=D:/the_code/runanywhere/SDK/rcli-meet/models/ggml-large-v3-turbo.bin"

cd /d "%~dp0"
echo rcli-translate — live translate the other person, then ask questions.
echo Commands: start / stop / save / load / add ^<path^> /quit
echo.

node src\quiet.js --minutes 20 %*

pause
