$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python bootstrap.py @args
