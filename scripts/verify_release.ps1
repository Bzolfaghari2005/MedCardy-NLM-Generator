$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Host "`n==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Set-Location (Split-Path -Parent $PSScriptRoot)

Invoke-Step "Python version" { python --version }
Invoke-Step "Project dependency resolution" {
    python -m pip install --dry-run -r requirements.txt
}
Invoke-Step "Production imports" {
    python -c "import fitz, streamlit, notebooklm, docx, markdown_it, openai, dotenv, charset_normalizer, pptx, openpyxl, bs4, filetype; print('Production imports OK')"
}
Invoke-Step "FFmpeg availability" { ffmpeg -version }
Invoke-Step "Python syntax" {
    python -m compileall -q -x "([\\/]\.git[\\/]|[\\/]data[\\/])" .
}
Invoke-Step "Parallel audio pipeline" { python test_parallel.py }
Invoke-Step "AI Folder" { python test_ai_folder.py }
Invoke-Step "Pytest suite" { python -m pytest tests -q }

Write-Host "`nRelease verification passed." -ForegroundColor Green
