$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$version = (Get-Content (Join-Path $root "VERSION") -Raw).Trim()
$dist = Join-Path $root "dist"
$stage = Join-Path ([System.IO.Path]::GetTempPath()) "nlm-source-$version"
$archive = Join-Path $dist "nlm-$version-source.zip"

Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $stage -Force | Out-Null
New-Item -ItemType Directory -Path $dist -Force | Out-Null

$rootFiles = @(
    ".env.example", ".gitignore", "CHANGELOG.md", "LICENSE",
    "README.md", "RELEASE_CHECKLIST.md", "VERSION",
    "requirements.txt", "requirements-dev.txt"
)
foreach ($name in $rootFiles) {
    Copy-Item (Join-Path $root $name) $stage
}
Copy-Item (Join-Path $root "*.py") $stage
Copy-Item (Join-Path $root "tests") (Join-Path $stage "tests") -Recurse
Copy-Item (Join-Path $root "scripts") (Join-Path $stage "scripts") -Recurse

Get-ChildItem $stage -Directory -Filter "__pycache__" -Recurse |
    Remove-Item -Recurse -Force
Get-ChildItem $stage -File -Include "*.pyc", "*.pyo" -Recurse |
    Remove-Item -Force

Remove-Item $archive -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $archive
Remove-Item $stage -Recurse -Force

Write-Host "Created $archive" -ForegroundColor Green
