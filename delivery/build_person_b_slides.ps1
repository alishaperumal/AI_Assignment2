param([string]$OutputName = '')
$ErrorActionPreference = 'Stop'
# Bundled runtime paths used for this local preparation. This installs nothing.
$runtimeRoot = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1026-487-0_26.1026.487.0_x64__3k8sg7r9htsxt\dependencies'
$env:ARTIFACT_NODE_MODULES = Join-Path $runtimeRoot 'node\node_modules'
$env:ARTIFACT_PYTHON = Join-Path $runtimeRoot 'python\python.exe'
$env:RUNTIME_NODE_MODULES = $env:ARTIFACT_NODE_MODULES
$env:RUNTIME_PYTHON = $env:ARTIFACT_PYTHON
$env:PRESENTATION_SKILL_DIR = 'C:\Users\user-pc\.codex\plugins\cache\openai-primary-runtime\presentations\26.1026.10807\skills\presentations'
$nodeExecutable = Join-Path $runtimeRoot 'node\bin\node.exe'
$env:RUNTIME_NODE = $nodeExecutable
$builder = Join-Path $PSScriptRoot 'build_person_b_slides.mjs'
if ($OutputName) {
    & $nodeExecutable $builder --output $OutputName
} else {
    & $nodeExecutable $builder
}
if ($LASTEXITCODE -ne 0) { throw "Slide builder failed with exit code $LASTEXITCODE" }

