param(
    [string]$OutputPath = ""
)

$releaseRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$reviewRoot = $env:MSF_SPI_REVIEW_SOURCE_ROOT
$workRoot = $env:MSF_SPI_WORK_SOURCE_ROOT
if ([string]::IsNullOrWhiteSpace($reviewRoot) -or [string]::IsNullOrWhiteSpace($workRoot)) {
    throw "Set MSF_SPI_REVIEW_SOURCE_ROOT and MSF_SPI_WORK_SOURCE_ROOT before rebuilding the private provenance manifest."
}
$robustnessRoot = Join-Path $workRoot "MSF_SPI_waveoptics_robustness"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $releaseRoot "docs\source_manifest.json"
}

$selected = @(
    @{ source = (Join-Path $workRoot "Moire_SPI_260603_sparse_response.py"); target = "src/msf_spi/ideal/simulation.py" },
    @{ source = (Join-Path $workRoot "optimize_coefficients.py"); target = "src/msf_spi/calibration/ideal_global_weights.py" },
    @{ source = (Join-Path $workRoot "Optimal_angle_diffs_SPI_circle_radial_stripes_optimized.json"); target = "data/angle_library/optimal_angle_differences.json" },
    @{ source = (Join-Path $workRoot "Test_image\USAF-1951.jpg"); target = "data/targets/USAF-1951.jpg" },
    @{ source = (Join-Path $robustnessRoot "metrics.py"); target = "src/msf_spi/metrics.py" },
    @{ source = (Join-Path $robustnessRoot "physical_bandwidth_config.py"); target = "src/msf_spi/physical/config.py" },
    @{ source = (Join-Path $robustnessRoot "physical_order_model.py"); target = "src/msf_spi/physical/order_model.py" },
    @{ source = (Join-Path $robustnessRoot "physical_pattern_cache.py"); target = "src/msf_spi/physical/pattern_cache.py" },
    @{ source = (Join-Path $robustnessRoot "physical_bandwidth_256.py"); target = "src/msf_spi/physical/bandwidth.py" },
    @{ source = (Join-Path $robustnessRoot "physical_msf_spi.py"); target = "src/msf_spi/physical/msf_patterns.py" },
    @{ source = (Join-Path $robustnessRoot "physical_fmax_msf_experiment.py"); target = "src/msf_spi/physical/comparison.py" },
    @{ source = (Join-Path $robustnessRoot "physical_global_calibration.py"); target = "src/msf_spi/calibration/global_weights.py" },
    @{ source = (Join-Path $robustnessRoot "physical_response_matrix.py"); target = "src/msf_spi/calibration/sparse_response.py" },
    @{ source = (Join-Path $robustnessRoot "physical_robustness_256.py"); target = "src/msf_spi/robustness/cases.py" },
    @{ source = (Join-Path $robustnessRoot "physical_robustness_256_pipeline.py"); target = "src/msf_spi/robustness/pipeline.py" },
    @{ source = (Join-Path $robustnessRoot "physical_robustness_256_artifacts.py"); target = "src/msf_spi/robustness/artifacts.py" },
    @{ source = (Join-Path $robustnessRoot "mode4_gpu.py"); target = "src/msf_spi/ideal/patterns.py" },
    @{ source = (Join-Path $robustnessRoot "mode4_sparse_response.py"); target = "src/msf_spi/calibration/sparse_response.py" },
    @{ source = (Join-Path $robustnessRoot "ideal_mode4_dual_angle.py"); target = "results/reference/angular_quantization/summary.json" },
    @{ source = (Join-Path $robustnessRoot "results_physical_robustness_256\nominal_response_matrix.npz"); target = "data/precomputed/physical_256_response_matrix.npz" },
    @{ source = (Join-Path $robustnessRoot "results_physical_robustness_256\global_weights.json"); target = "data/precomputed/physical_256_global_weights.json" },
    @{ source = (Join-Path $robustnessRoot "results_ideal_mode4_radial_stripes_shared_calibration\calibration_0p001deg\response_matrix.npz"); target = "data/precomputed/ideal_response_matrix_0p001deg.npz" },
    @{ source = (Join-Path $robustnessRoot "results_ideal_mode4_radial_stripes_shared_calibration\calibration_0p001deg\response_matrix_responses.npz"); target = "data/precomputed/ideal_response_matrix_0p001deg_responses.npz" },
    @{ source = (Join-Path $robustnessRoot "results_ideal_mode4_radial_stripes_shared_calibration\calibration_0p001deg\response_matrix_metadata.json"); target = "data/precomputed/ideal_response_matrix_0p001deg_metadata.json" },
    @{ source = (Join-Path $robustnessRoot "results_ideal_mode4_radial_stripes_shared_calibration\dual_angle_comparison.png"); target = "results/reference/angular_quantization/comparison.png" },
    @{ source = (Join-Path $robustnessRoot "results_ideal_mode4_radial_stripes_shared_calibration\dual_angle_summary.json"); target = "results/reference/angular_quantization/summary.json" },
    @{ source = (Join-Path $robustnessRoot "results_physical_robustness_256\summary_metrics.json"); target = "results/reference/robustness/summary_metrics.json" },
    @{ source = (Join-Path $robustnessRoot "results_physical_robustness_256\configuration.json"); target = "results/reference/robustness/configuration.json" },
    @{ source = (Join-Path $reviewRoot "R5.png"); target = "results/reference/physical_comparison/Figure_R5.png" },
    @{ source = (Join-Path $reviewRoot "R6.png"); target = "results/reference/robustness/Figure_R6.png" }
)

$entries = foreach ($item in $selected) {
    if (-not (Test-Path -LiteralPath $item.source -PathType Leaf)) {
        throw "Required source file is missing: $($item.source)"
    }
    $file = Get-Item -LiteralPath $item.source
    [ordered]@{
        source = $file.FullName
        target = $item.target
        bytes = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$payload = [ordered]@{
    schema_version = 1
    generated_utc = [DateTime]::UtcNow.ToString("o")
    entries = @($entries)
}

$json = $payload | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText($OutputPath, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
Write-Host "Wrote $($entries.Count) source records to $OutputPath"
