# Sequential Person C GPU queue (run from the repository root). Each step starts
# only after the previous one finishes, so GPU jobs never overlap and timed
# benchmarks run without other heavy work. Failures are detected by exit code
# (stderr warnings must not abort a step in Windows PowerShell 5.1).
$py = ".\.venv\Scripts\python.exe"
$logs = "runs\person_c\logs"
$teacher = "runs/person_b_final/best_weights.pt"
$base = "configs/person_c/seed45_selected.yaml"
$final = "configs/person_c/final_postprocess.yaml"
$c = "runs/person_c/compression"

function Step($name, [scriptblock]$block) {
    $t0 = Get-Date
    Write-Output "[$(Get-Date -Format HH:mm:ss)] start $name"
    & $block *> "$logs\$name.log"
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE (see $logs\$name.log)" }
    Write-Output "[$(Get-Date -Format HH:mm:ss)] done $name in $([int]((Get-Date)-$t0).TotalSeconds) s"
}

while (-not (Test-Path "$c/kd_resnet18/summary.json")) { Start-Sleep -Seconds 20 }
Write-Output "KD student finished"

Step "label_resnet18" { & $py compress.py distill --teacher $teacher --config $base --output "$c/label_resnet18" --name label_resnet18_seed45 --encoder resnet18 --kd-weight 0 --validate-from 11 }
Step "pruned50" { & $py compress.py prune --teacher $teacher --config $base --output "$c/pruned50" --name pruned50_seed45 --ratio 0.5 --epochs 10 --learning-rate 0.0001 --validate-from 1 }
Step "teacher_fp16" { & $py compress.py fp16 --teacher $teacher --output "$c/teacher_fp16.pt" }

$variants = @(
    @("teacher_amp16", $teacher, "amp16"),
    @("teacher_fp32", $teacher, "fp32"),
    @("teacher_fp16", "$c/teacher_fp16.pt", "fp16"),
    @("kd_resnet18", "$c/kd_resnet18/best_weights.pt", "amp16"),
    @("label_resnet18", "$c/label_resnet18/best_weights.pt", "amp16"),
    @("pruned50", "$c/pruned50/best_weights.pt", "amp16")
)
foreach ($v in $variants) {
    Step "bench_$($v[0])" { & $py compress.py benchmark --variant $v[1] --name $v[0] --config $final --output "$c/bench_$($v[0])" --precision $v[2] --save-predictions }
}
foreach ($v in $variants) {
    Step "pp_check_$($v[0])" { & $py offline_eval.py sweep --predictions "$c/bench_$($v[0])/validation/predictions" --config $base --grid configs/person_c/pp_check.yaml --output "$c/pp_check_$($v[0])" --workers 8 --bootstrap 1000 }
}
Write-Output "[$(Get-Date -Format HH:mm:ss)] queue complete"
