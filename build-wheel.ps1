param(
    [string]$PythonExe = "python",
    [switch]$IncludeTorchVersion,
    [switch]$IncludeCudaVersion
)

$ErrorActionPreference = "Stop"

& $PythonExe -m pip install --upgrade build
& $PythonExe -m build --wheel

$wheel = Get-ChildItem -Path (Join-Path $PSScriptRoot "dist") -Filter "neurcross-*.whl" |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

if (-not $wheel) {
    throw "No wheel was produced in dist\."
}

if (-not $IncludeTorchVersion -and -not $IncludeCudaVersion) {
    Write-Host "Built wheel: $($wheel.Name)"
    return
}

$torchInfoJson = & $PythonExe -c @"
import json

try:
    import torch
except Exception:
    print(json.dumps({"torch": "", "cuda": ""}))
else:
    torch_version = getattr(torch, "__version__", "") or ""
    cuda_version = getattr(torch.version, "cuda", "") or ""
    if "+" in torch_version:
        torch_version = torch_version.split("+", 1)[0]
    print(json.dumps({"torch": torch_version, "cuda": cuda_version}))
"@

$torchInfo = $torchInfoJson | ConvertFrom-Json
$buildParts = @()

if ($IncludeTorchVersion -and $torchInfo.torch) {
    $torchToken = ($torchInfo.torch.ToLower() -replace '[^0-9a-z]', '')
    if ($torchToken) {
        $buildParts += "torch$torchToken"
    }
}

if ($IncludeCudaVersion -and $torchInfo.cuda) {
    $cudaToken = ($torchInfo.cuda.ToLower() -replace '[^0-9a-z]', '')
    if ($cudaToken) {
        $buildParts += "cu$cudaToken"
    }
}

if ($buildParts.Count -eq 0) {
    Write-Host "Built wheel: $($wheel.Name)"
    return
}

$match = [regex]::Match($wheel.Name, '^(?<dist>.+)-(?<version>[^-]+)-(?<py>[^-]+)-(?<abi>[^-]+)-(?<plat>[^.]+)\.whl$')
if (-not $match.Success) {
    throw "Unexpected wheel filename format: $($wheel.Name)"
}

$buildTag = "1" + ($buildParts -join "")
$renamedWheel = "{0}-{1}-{2}-{3}-{4}-{5}.whl" -f `
    $match.Groups["dist"].Value, `
    $match.Groups["version"].Value, `
    $buildTag, `
    $match.Groups["py"].Value, `
    $match.Groups["abi"].Value, `
    $match.Groups["plat"].Value

$renamedPath = Join-Path $wheel.DirectoryName $renamedWheel
Move-Item -LiteralPath $wheel.FullName -Destination $renamedPath -Force
Write-Host "Built wheel: $renamedWheel"
