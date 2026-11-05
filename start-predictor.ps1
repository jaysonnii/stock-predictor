# Starts the stock predictor (if it is not already running) and opens it in your browser.
# The daily $1M check in appears as soon as the page opens.
# Set up to run at logon and every morning with the Register-ScheduledTask command in README.

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

$python = @(".venv\Scripts\python.exe", "venv\Scripts\python.exe", "env\Scripts\python.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    Write-Host "No virtual environment found in $repo (looked for .venv, venv, env)."
    exit 1
}

function Test-Server {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", 8000)
        $client.Close()
        return $true
    } catch { return $false }
}

if (-not (Test-Server)) {
    Start-Process -WindowStyle Hidden -FilePath (Resolve-Path $python) `
        -ArgumentList "-m", "uvicorn", "main:app", "--port", "8000" -WorkingDirectory $repo
    for ($i = 0; $i -lt 20 -and -not (Test-Server); $i++) { Start-Sleep -Milliseconds 500 }
}

Start-Process "http://127.0.0.1:8000"
