# Windows (PowerShell): run ECIP locally without Docker. Requires Python 3.10+ and Ollama.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Get-Content .env | Where-Object { $_ -match '^\s*[A-Z_]+=' } | ForEach-Object {
  $k, $v = $_.Split('=', 2); $v = ($v -split '\s+#')[0].Trim()
  if (-not [Environment]::GetEnvironmentVariable($k)) { [Environment]::SetEnvironmentVariable($k, $v) }
}
if (-not (Test-Path .venv)) {
  python -m venv .venv
  .\.venv\Scripts\python -m pip install -U pip
  .\.venv\Scripts\python -m pip install -r requirements.txt
}
$llm = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "llama3.2:3b" }
$emb = if ($env:EMBED_MODEL) { $env:EMBED_MODEL } else { "nomic-embed-text" }
if (($env:LLM_PROVIDER -eq $null) -or ($env:LLM_PROVIDER -eq "ollama")) { ollama pull $llm; ollama pull $emb }
Write-Host "ECIP on http://localhost:8000"
.\.venv\Scripts\python -m api.main
