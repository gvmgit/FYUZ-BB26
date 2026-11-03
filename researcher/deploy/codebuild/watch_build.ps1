# Follow a CodeBuild build until it finishes, then print the tail of its log.
#   .\deploy\codebuild\watch_build.ps1 -BuildId atr-deploy:<uuid>
param([Parameter(Mandatory = $true)][string]$BuildId, [int]$PollSeconds = 60, [int]$MaxMinutes = 0)
$deadline = if ($MaxMinutes -gt 0) { (Get-Date).AddMinutes($MaxMinutes) } else { [datetime]::MaxValue }
$env:AWS_CA_BUNDLE = "$env:USERPROFILE\.aws\ca-bundle.pem"
$aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$region = if ($env:ATR_DEPLOY_REGION) { $env:ATR_DEPLOY_REGION } else { "us-east-2" }
$common = @("--profile", "atr", "--region", $region)
$last = ""
while ($true) {
    $b = & $aws codebuild batch-get-builds --ids $BuildId @common --query "builds[0].[buildStatus,currentPhase]" --output text
    if ($b -ne $last) { "$(Get-Date -Format HH:mm:ss)  $b"; $last = $b }
    if ($b -notmatch "IN_PROGRESS") { break }
    if ((Get-Date) -gt $deadline) { "STILL RUNNING after $MaxMinutes min (phase: $b)"; exit 0 }
    Start-Sleep $PollSeconds
}
$stream = $BuildId.Split(":")[1]
"----- last log lines -----"
& $aws logs get-log-events --log-group-name /atr/codebuild --log-stream-name $stream --no-start-from-head --limit 120 @common --query "events[].message" --output text
