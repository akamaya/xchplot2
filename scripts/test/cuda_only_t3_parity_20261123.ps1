# Reproducible CUDA-only vs verified hybrid T3 parity experiment.
# Run from a VS2022 Developer PowerShell after building both executables.
# No OFF benchmarks. The reference .gplot is never overwritten.
param(
    [string]$OutputDir = '\\NAS-UGREEN\personal_folder\plots_test\cuda_only_parity'
)
$ErrorActionPreference = 'Stop'
$repo = 'C:\dev\xchplot2'
$native = Join-Path $repo 'build\windows-native-cuda-only-nvcc\native_cuda_pipeline.exe'
$hybrid = Join-Path $repo 'build\windows-cuda-core\tools\xchplot2\xchplot2.exe'
$gid = '0' * 64
$expectedId = '0d5535e13cc9708d0ff0289af2fae27e564b6bcbcd9242f5140d96957744a517'
$filename = 'cuda_group_zero_parity_20261123.gplot'
$plot = Join-Path $OutputDir $filename
Set-Location $repo
if ((git branch --show-current) -ne 'cuda/nvidia-12gb-k28s6') {
    throw 'Wrong Git branch; expected cuda/nvidia-12gb-k28s6'
}
if (!(Test-Path -LiteralPath $native) -or !(Test-Path -LiteralPath $hybrid)) {
    throw 'Native CUDA and hybrid xchplot2 executables must be built first'
}
if (Test-Path -LiteralPath $plot) {
    throw "Refusing to overwrite a preexisting reference plot: $plot"
}
Write-Host '[parity] Checking group ID derivation without large GPU allocation'
$plan = (& $native --plan --group-id $gid | Out-String)
if ($LASTEXITCODE -ne 0) { throw 'Native CUDA plan failed' }
if ($plan -notmatch [regex]::Escape($expectedId)) {
    throw 'Native CUDA Plot ID derivation differs from upstream SHA256'
}
Write-Host $plan
Write-Host '[parity] Running native CUDA-only full compute'
$nativeText = (& $native --run --group-id $gid | Out-String)
if ($LASTEXITCODE -ne 0) { throw 'Native CUDA full compute failed' }
Write-Host $nativeText
$nativeMatch = [regex]::Match($nativeText, 'canonical unique T3 fragments=(\d+) SHA256=([a-fA-F0-9]{64})')
if (!$nativeMatch.Success) {
    throw 'Native CUDA output lacks a canonical unique T3 count / SHA256'
}
$nativeCount = [uint64]::Parse($nativeMatch.Groups[1].Value)
$nativeHash = $nativeMatch.Groups[2].Value.ToLowerInvariant()
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Write-Host '[parity] Writing hybrid reference plot using the identical Group ID'
& $hybrid test $gid 0 0 0 -G -o $OutputDir -N $filename
if ($LASTEXITCODE -ne 0) { throw 'Hybrid reference plotting failed' }
if (!(Test-Path -LiteralPath $plot)) { throw "Hybrid output missing: $plot" }
Write-Host '[parity] Decoding reference plot fragments and hashing'
$reference = (& $hybrid fragments-hash $plot | Out-String)
if ($LASTEXITCODE -ne 0) { throw 'Reference fragment decoding/hash failed' }
Write-Host $reference
$countMatch = [regex]::Match($reference, 'decoded fragments=(\d+)')
$hashMatch = [regex]::Match($reference, 'sorted decoded T3 SHA256=([a-fA-F0-9]{64})')
if (!$countMatch.Success -or !$hashMatch.Success) {
    throw 'Reference output lacks T3 count / SHA256'
}
$referenceCount = [uint64]::Parse($countMatch.Groups[1].Value)
$referenceHash = $hashMatch.Groups[1].Value.ToLowerInvariant()
Write-Host "[parity] native unique=$nativeCount sha256=$nativeHash"
Write-Host "[parity] decoded gplot=$referenceCount sha256=$referenceHash"
if ($nativeCount -ne $referenceCount -or $nativeHash -ne $referenceHash) {
    throw 'CUDA-only and hybrid canonical T3 fragments do not match'
}
Write-Host '[parity] Canonical T3 count and SHA256 MATCH'
Write-Host '[parity] Running sampled Full Proof validation on hybrid reference plot'
& $hybrid verify $plot --full --trials 100
if ($LASTEXITCODE -ne 0) { throw 'Hybrid full-proof validation failed' }
Write-Host '[parity] PASS: same-ID T3 hash parity + sampled hybrid Full Proof'
Write-Host "[parity] Reference retained: $plot"
