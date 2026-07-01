$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python trellis2_bootstrap.py @args
