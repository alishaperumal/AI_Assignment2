# CPU deployment queue: starts only after the GPU queue has finished, so CPU
# load never overlaps the timed GPU benchmarks. Run from the repository root.
$py = ".\.venv\Scripts\python.exe"
$logs = "runs\person_c\logs"
$teacher = "runs/person_b_final/best_weights.pt"
$base = "configs/person_c/seed45_selected.yaml"
$final = "configs/person_c/final_postprocess.yaml"
$c = "runs/person_c/compression"
$subset = 20

function Step($name, [scriptblock]$block) {
    $t0 = Get-Date
    Write-Output "[$(Get-Date -Format HH:mm:ss)] start $name"
    & $block *> "$logs\$name.log"
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE (see $logs\$name.log)" }
    Write-Output "[$(Get-Date -Format HH:mm:ss)] done $name in $([int]((Get-Date)-$t0).TotalSeconds) s"
}

while (-not (Test-Path "$c/pp_check_pruned50/summary.json")) { Start-Sleep -Seconds 20 }
Start-Sleep -Seconds 5
Write-Output "GPU queue finished"

# Declared follow-up after observing that 50% uniform pruning collapsed (val PQ 0 in early
# fine-tuning epochs): a milder 30% pruning fine-tuned at the original learning rate.
Step "pruned30" { & $py compress.py prune --teacher $teacher --config $base --output "$c/pruned30" --name pruned30_seed45 --ratio 0.3 --epochs 10 --learning-rate 0.0003 --validate-from 1 }
Step "bench_pruned30" { & $py compress.py benchmark --variant "$c/pruned30/best_weights.pt" --name pruned30 --config $final --output "$c/bench_pruned30" --precision amp16 --save-predictions }
Step "pp_check_pruned30" { & $py offline_eval.py sweep --predictions "$c/bench_pruned30/validation/predictions" --config $base --grid configs/person_c/pp_check.yaml --output "$c/pp_check_pruned30" --workers 8 --bootstrap 1000 }

Step "onnx_export" { & $py compress_onnx.py export --teacher $teacher --config $base --output "$c/teacher_fp32.onnx" }
Step "onnx_int8" { & $py compress_onnx.py int8 --onnx "$c/teacher_fp32.onnx" --output "$c/teacher_int8.onnx" --calibration-images 8 --crop 1024 }
Step "bench_cpu_pytorch_fp32" { & $py compress.py benchmark --variant $teacher --name cpu_pytorch_fp32 --config $final --output "$c/bench_cpu_pytorch_fp32" --precision fp32 --device cpu --skip-validation --test-subset $subset }
Step "bench_cpu_onnx_fp32" { & $py compress.py benchmark --variant "$c/teacher_fp32.onnx" --name cpu_onnx_fp32 --config $final --output "$c/bench_cpu_onnx_fp32" --precision fp32 --device cpu --skip-validation --test-subset $subset }
Step "bench_cpu_onnx_int8" { & $py compress.py benchmark --variant "$c/teacher_int8.onnx" --name cpu_onnx_int8 --config $final --output "$c/bench_cpu_onnx_int8" --precision fp32 --device cpu --save-predictions --test-subset $subset }
Step "pp_check_cpu_onnx_int8" { & $py offline_eval.py sweep --predictions "$c/bench_cpu_onnx_int8/validation/predictions" --config $base --grid configs/person_c/pp_check.yaml --output "$c/pp_check_cpu_onnx_int8" --workers 8 --bootstrap 1000 }
Write-Output "[$(Get-Date -Format HH:mm:ss)] CPU queue complete"

