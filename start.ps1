[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$NoSolutions,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$JupyterArguments
)

$ErrorActionPreference = 'Stop'
$environmentPath = Join-Path $PSScriptRoot '.venv'
$pythonPath = Join-Path $environmentPath 'python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project environment is missing. See README.md for setup commands.'
}

$previousPath = $env:PATH
Push-Location -LiteralPath $PSScriptRoot
try {
    # Include Conda DLLs and FFmpeg even from a terminal without Conda activation.
    $environmentPaths = @(
        $environmentPath
        (Join-Path $environmentPath 'Library\mingw-w64\bin')
        (Join-Path $environmentPath 'Library\usr\bin')
        (Join-Path $environmentPath 'Library\bin')
        (Join-Path $environmentPath 'Scripts')
        (Join-Path $environmentPath 'bin')
    )
    $env:PATH = ($environmentPaths -join [IO.Path]::PathSeparator) + [IO.Path]::PathSeparator + $previousPath

    if (-not $NoBrowser -and -not $NoSolutions) {
        try {
            Start-Process -FilePath 'https://github.com/eje24/iap-diffusion-labs/tree/2026/solutions'
        }
        catch {
            Write-Warning "Could not open the solutions page: $($_.Exception.Message)"
        }
    }

    $openBrowser = if ($NoBrowser) { 'False' } else { 'True' }
    $launchArguments = @(
        '-m', 'jupyterlab'
        "--ServerApp.open_browser=$openBrowser"
        '--ServerApp.answer_yes=True'
    )

    Write-Host 'Starting JupyterLab. Keep this terminal open; press Ctrl+C here to stop it.'
    # Run in the foreground so Ctrl+C reaches Jupyter without a Conda batch wrapper.
    & $pythonPath @launchArguments @JupyterArguments
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 130 -and $LASTEXITCODE -ne -1073741510) {
        throw "JupyterLab exited with code $LASTEXITCODE."
    }
}
finally {
    $env:PATH = $previousPath
    Pop-Location
}
