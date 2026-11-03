# Live dashboard (http://localhost:8770) for the us-east-2 deployment.
Set-Location (Split-Path $PSScriptRoot)
$env:ATR_DEPLOY_REGION = "us-east-2"
. .\deploy\aws_env.ps1
$env:AWS_REGION = "us-east-2"; $env:AWS_DEFAULT_REGION = "us-east-2"
..\.venv\Scripts\python.exe deploy\live_dashboard.py
