# Second follow-up (quiet window, nothing else running):
# 1. pruned-30% model stored in FP16 (composition of two evaluated techniques; added after
#    the first results, evaluated under the unchanged final-model rule);
# 2. CPU variants re-timed, because the first CPU timings overlapped slide rendering.
#    The first CPU runs are kept in compression/superseded/ for the record.
$py = ".\.venv\Scripts\python.exe"
$logs = "runs\person_c\logs"
$base = "configs/person_c/seed45_selected.yaml"
$final = "configs/person_c/final_postprocess.yaml"
$c = "runs/person_c/compression"
$teacher = "runs/person_b_final/best_weights.pt"
$subset = 20

function Step($name, [scriptblock]$block) {
    $t0 = Get-Date
    Write-Output "[$(Get-Date -Format HH:mm:ss)] start $name"
    & $block *> "$logs\$name.log"
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE (see $logs\$name.log)" }
    Write-Output "[$(Get-Date -Format HH:mm:ss)] done $name in $([int]((Get-Date)-$t0).TotalSeconds) s"
}

Step "pruned30_fp16" { & $py compress.py fp16 --teacher "$c/pruned30/best_weights.pt" --output "$c/pruned30_fp16.pt" }
Step "bench_pruned30_fp16" { & $py compress.py benchmark --variant "$c/pruned30_fp16.pt" --name pruned30_fp16 --config $final --output "$c/bench_pruned30_fp16" --precision fp16 --save-predictions }
Step "pp_check_pruned30_fp16" { & $py offline_eval.py sweep --predictions "$c/bench_pruned30_fp16/validation/predictions" --config $base --grid configs/person_c/pp_check.yaml --output "$c/pp_check_pruned30_fp16" --workers 8 --bootstrap 1000 }

New-Item -ItemType Directory -Force "$c/superseded" | Out-Null
foreach ($d in "bench_cpu_pytorch_fp32", "bench_cpu_onnx_fp32", "bench_cpu_onnx_int8", "pp_check_cpu_onnx_int8") {
    Move-Item "$c/$d" "$c/superseded/${d}_overlapped_rendering"
}
Step "bench_cpu_pytorch_fp32_quiet" { & $py compress.py benchmark --variant $teacher --name cpu_pytorch_fp32 --config $final --output "$c/bench_cpu_pytorch_fp32" --precision fp32 --device cpu --skip-validation --test-subset $subset }
Step "bench_cpu_onnx_fp32_quiet" { & $py compress.py benchmark --variant "$c/teacher_fp32.onnx" --name cpu_onnx_fp32 --config $final --output "$c/bench_cpu_onnx_fp32" --precision fp32 --device cpu --skip-validation --test-subset $subset }
Step "bench_cpu_onnx_int8_quiet" { & $py compress.py benchmark --variant "$c/teacher_int8.onnx" --name cpu_onnx_int8 --config $final --output "$c/bench_cpu_onnx_int8" --precision fp32 --device cpu --save-predictions --test-subset $subset }
Step "pp_check_cpu_onnx_int8_quiet" { & $py offline_eval.py sweep --predictions "$c/bench_cpu_onnx_int8/validation/predictions" --config $base --grid configs/person_c/pp_check.yaml --output "$c/pp_check_cpu_onnx_int8" --workers 8 --bootstrap 1000 }
Write-Output "[$(Get-Date -Format HH:mm:ss)] follow-up complete"
