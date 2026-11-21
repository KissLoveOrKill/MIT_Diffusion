[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project environment is missing. See README.md for setup commands.'
}

# List servers without relying on the original console or Conda activation.
$serverLines = @(& $pythonPath -m jupyter server list --json)
if ($LASTEXITCODE -ne 0) {
    throw 'Could not list running Jupyter servers.'
}

$projectPath = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
$found = $false
foreach ($line in $serverLines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $server = $line | ConvertFrom-Json
    if (-not $server.root_dir) { continue }
    $serverRoot = [IO.Path]::GetFullPath([string]$server.root_dir).TrimEnd('\', '/')
    if ($serverRoot -ne $projectPath) { continue }

    $serverProcess = Get-Process -Id ([int]$server.pid) -ErrorAction SilentlyContinue
    if (-not $serverProcess) { continue }
    $found = $true
    try {
        Write-Host "Stopping MIT Diffusion Jupyter on port $($server.port)..."
        try {
            $shutdownUri = [Uri]::new([Uri]$server.url, 'api/shutdown')
            $headers = @{ Authorization = 'token ' + $server.token }
            Invoke-RestMethod -Uri $shutdownUri -Method Post -Headers $headers -Body '' -TimeoutSec 5 | Out-Null
            if (-not $serverProcess.WaitForExit(15000)) {
                throw 'Jupyter did not exit within 15 seconds.'
            }
        }
        catch {
            if (-not $Force) {
                throw 'Jupyter could not shut down normally. Retry with .\stop.ps1 -Force.'
            }
            # A forced stop is limited to the project Python verified above.
            if (-not $serverProcess.HasExited) {
                if ($serverProcess.Path -ne $pythonPath) {
                    throw 'Refusing to force-stop a process outside the project environment.'
                }
                Stop-Process -InputObject $serverProcess -Force
                if (-not $serverProcess.WaitForExit(5000)) {
                    throw 'The project Jupyter process did not exit.'
                }
            }
        }
        Write-Host 'Stopped.'
    }
    finally {
        $serverProcess.Dispose()
    }
}
if (-not $found) {
    Write-Host 'No running Jupyter server found for MIT Diffusion.'
}
