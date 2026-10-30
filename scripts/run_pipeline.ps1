param(
    [string]$DataDir = "research_intern_coding_test",
    [ValidateSet("bag", "transformer")][string]$Model = "transformer"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Project interpreter missing. Create .venv with Python 3.12 and install dependencies as documented in README.md."
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonExecutable -m ehr_forecast prepare --data-dir $DataDir --max-events 256
    if ($LASTEXITCODE -ne 0) { throw "Data preparation failed" }
    & $pythonExecutable -m ehr_forecast train --config "configs/$Model.json"
    if ($LASTEXITCODE -ne 0) { throw "Training failed (use a new output_dir for another experiment)" }
    $configuration = Get-Content -LiteralPath "configs/$Model.json" -Raw | ConvertFrom-Json
    $checkpointPath = Join-Path $configuration.output_dir 'best.pt'
    & $pythonExecutable -m ehr_forecast predict --checkpoint $checkpointPath --output predictions.csv
    if ($LASTEXITCODE -ne 0) { throw "Prediction failed" }
    & $pythonExecutable -m ehr_forecast validate-submission --data-dir $DataDir --predictions predictions.csv
    if ($LASTEXITCODE -ne 0) { throw "Submission validation failed" }
} finally {
    Pop-Location
}
