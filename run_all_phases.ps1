$ErrorActionPreference = "Continue"

Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path outputs | Out-Null
    $report = @()
    $report += "# HW8 Phase Completion Report"
    $report += ""
    $report += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $report += ""

    $report += "## Phase 1 - Manim Concept Animation"
    python -c "import manim; print(manim.__version__)" *> outputs/phase1_manim_check.log
    if ($LASTEXITCODE -eq 0) {
        manim -ql animations/svm_manim.py LinearSVMMarginScene *> outputs/phase1_linear_manim.log
        $linearExit = $LASTEXITCODE
        manim -ql animations/svm_manim.py KernelTrick3DScene *> outputs/phase1_kernel_manim.log
        $kernelExit = $LASTEXITCODE
        if ($linearExit -eq 0 -and $kernelExit -eq 0) {
            $report += "- Status: rendered successfully."
        }
        else {
            $report += "- Status: Manim installed, but rendering failed. See outputs/phase1_*_manim.log."
        }
    }
    else {
        $report += "- Status: script completed, render blocked in this environment."
        $report += "- Reason: Manim is not installed; on Python 3.14, moderngl/glcontext require Microsoft C++ Build Tools."
        $report += "- Source: animations/svm_manim.py"
    }
    $report += ""

    $report += "## Phase 2 - Scikit-Learn True Decision Surface"
    python scripts/export_phase2.py *> outputs/phase2_export.log
    if ($LASTEXITCODE -eq 0) {
        $report += "- Status: completed."
        $report += "- Outputs:"
        $report += "  - outputs/phase2_decision_boundary.html"
        $report += "  - outputs/phase2_kernel_lift_3d.html"
        $report += "  - outputs/phase2_metrics.json"
    }
    else {
        $report += "- Status: failed. See outputs/phase2_export.log."
    }
    $report += ""

    $report += "## Phase 3 - Streamlit / Plotly Dashboard"
    try {
        $status = (Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 3).StatusCode
    }
    catch {
        Start-Process -FilePath python -ArgumentList "-m","streamlit","run","app/streamlit_app.py","--server.port","8501","--server.headless","true" -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
        Start-Sleep -Seconds 4
        try {
            $status = (Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 5).StatusCode
        }
        catch {
            $status = "failed"
        }
    }
    if ($status -eq 200) {
        $report += "- Status: completed and reachable."
        $report += "- URL: http://localhost:8501"
    }
    else {
        $report += "- Status: failed to confirm Streamlit server."
    }
    $report += ""

    $report += "## Validation"
    python -m compileall src app animations *> outputs/compileall.log
    $compileExit = $LASTEXITCODE
    python -m pytest -q *> outputs/pytest.log
    $pytestExit = $LASTEXITCODE
    if ($compileExit -eq 0 -and $pytestExit -eq 0) {
        $report += "- compileall: passed"
        $report += "- pytest: passed"
    }
    else {
        $report += "- compileall or pytest failed. See outputs/compileall.log and outputs/pytest.log."
    }

    $report | Set-Content -LiteralPath outputs/phase_completion_report.md -Encoding UTF8
    Get-Content -LiteralPath outputs/phase_completion_report.md
}
finally {
    Pop-Location
}
