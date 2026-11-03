# Review console (http://localhost:8765/console/) on the us-east-2 deployment: finding cards from DynamoDB, verdicts
# through the checkpoint (only a Human principal may ApproveChange).
Set-Location (Split-Path $PSScriptRoot)
$env:ATR_DEPLOY_REGION = "us-east-2"
. .\deploy\aws_env.ps1
$env:AWS_REGION = "us-east-2"; $env:AWS_DEFAULT_REGION = "us-east-2"
Get-Content deploy\cdk\atr_aws.env | Where-Object { $_ -match '^[A-Z_]+=' } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }
$env:ATR_GRAPH = "local"      # Neptune is reachable only inside the VPC; the laptop console uses the local graph
..\.venv\Scripts\python.exe -m atr.cli --mode aws serve --port 8765
