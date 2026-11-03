# Dot-source before running Python against AWS:  . .\deploy\aws_env.ps1
# Loads the short-lived credentials of the `atr` profile (from `aws login`) into this shell's environment, without printing
# them, and points every AWS client at the CA bundle built from the Windows trust store (corporate TLS inspection).
# Region: ATR_DEPLOY_REGION if set, else us-east-2.
$region = if ($env:ATR_DEPLOY_REGION) { $env:ATR_DEPLOY_REGION } else { "us-east-2" }
$env:AWS_CA_BUNDLE = "$env:USERPROFILE\.aws\ca-bundle.pem"
$env:AWS_REGION = $region
$env:AWS_DEFAULT_REGION = $region
$env:ATR_REGION = $region
$c = & "C:\Program Files\Amazon\AWSCLIV2\aws.exe" configure export-credentials --profile atr --format process | ConvertFrom-Json
$env:AWS_ACCESS_KEY_ID = $c.AccessKeyId
$env:AWS_SECRET_ACCESS_KEY = $c.SecretAccessKey
$env:AWS_SESSION_TOKEN = $c.SessionToken
Remove-Item Env:AWS_PROFILE -ErrorAction SilentlyContinue
"AWS session loaded for $region (expires $($c.Expiration))"