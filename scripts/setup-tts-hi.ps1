# Download Hindi Piper TTS for rcli-translate (sherpa-onnx pack).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$models = Join-Path $root 'models'
$dest = Join-Path $models 'vits-piper-hi_IN-pratham-medium'
$onnx = Join-Path $dest 'hi_IN-pratham-medium.onnx'
if (Test-Path $onnx) {
  Write-Host "Hindi TTS already present: $dest"
  exit 0
}
New-Item -ItemType Directory -Force -Path $models | Out-Null
$url = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-pratham-medium.tar.bz2'
$archive = Join-Path $models 'vits-piper-hi_IN-pratham-medium.tar.bz2'
Write-Host "Downloading Hindi TTS (~64 MB)..."
curl.exe -L --retry 3 -o $archive $url
if (-not (Test-Path $archive)) { throw "download failed" }
Write-Host "Extracting..."
tar -xjf $archive -C $models
Remove-Item $archive -Force -ErrorAction SilentlyContinue
if (-not (Test-Path $onnx)) { throw "extract failed — $onnx missing" }
Write-Host "OK: $dest"
