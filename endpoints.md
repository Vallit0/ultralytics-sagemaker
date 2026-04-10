```bash
aws sagemaker-runtime invoke-endpoint `
  --endpoint-name $ENDPOINT_NAME `
  --content-type "text/csv" `
  --body "3,1,25,0,0,7.25" `
  --cli-binary-format raw-in-base64-out `
  prediction.json
Get-Content prediction.json
```
Write-Host "---"
```bash
aws sagemaker-runtime invoke-endpoint `
  --endpoint-name $ENDPOINT_NAME `
  --content-type "text/csv" `
  --body "1,0,30,0,0,100.0" `
  --cli-binary-format raw-in-base64-out `
  prediction2.json
Get-Content prediction2.json
```