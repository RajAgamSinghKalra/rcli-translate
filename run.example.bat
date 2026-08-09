@echo off
setlocal

:: Copy this file to run.bat and fill in the paths for your machine.

:: Path to your @runanywhere/electron dist/ build.
set "RCLI_XL8_SDK_DIST=D:/the_code/runanywhere/SDK/runanywhere-sdks-main/sdk/runanywhere-electron/dist"

:: RunAnywhere LLM catalog id or local GGUF path.
set "RCLI_XL8_LLM_PATH=D:/the_code/gpu-bench-qwen3/Qwen3-4B-Q4_K_M.gguf"

:: Optional: real python.exe if `python` is the Windows Store stub.
set "RCLI_XL8_PYTHON=C:/Users/rajag/AppData/Local/Programs/Python/Python314/python.exe"

:: Your language - live translations go into this (Whisper auto-detects theirs).
set "RCLI_XL8_TO=hi"

:: Label for the other person in captions.
set "RCLI_XL8_OTHER=transcribing"

:: Optional: reuse Whisper assets from an existing rcli-meet install.
:: set "RCLI_XL8_WHISPER_BIN=D:/the_code/runanywhere/SDK/rcli-meet/bin/whisper"
:: set "RCLI_XL8_WHISPER_MODEL=D:/the_code/runanywhere/SDK/rcli-meet/models/ggml-large-v3-turbo.bin"

:: Hindi TTS (run once): powershell -ExecutionPolicy Bypass -File scripts\setup-tts-hi.ps1

:: Mute original voice - hear only translations (needs VB-Audio Cable).
:: set "RCLI_XL8_MUTE_ORIGINAL=1"
:: set "RCLI_XL8_LOOPBACK=CABLE"
:: set "RCLI_XL8_SPEAKERS=Kraken"

cd /d "%~dp0"
echo rcli-translate - live translate the other person, then ask questions.
echo Commands: start / stop / save / load / add ^<path^> /quit
echo.

node src\quiet.js --minutes 20 --from auto --pick-to %*

pause
