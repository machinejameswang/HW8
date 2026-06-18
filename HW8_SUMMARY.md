# HW8 Summary - SVM Kernel Trick 3D Interactive Demo

Date: 2026-06-18

## 1. Master Prompt

Created and cleaned:

- `svm_project_prompt.yaml`

Purpose:

- A complete 3-phase YAML prompt for Antigravity Agent.
- Includes Manim concept animation, Scikit-Learn SVM decision-surface computation, and Streamlit/Plotly interactive dashboard.
- Added the critical label rule: blue points must be center/core, red points must be outer ring.

Validation:

- YAML parsed successfully with `yaml.safe_load`.

## 2. Phase 1 - Manim Concept Animation

Implemented:

- `animations/svm_manim.py`
- `LinearSVMMarginScene`
- `KernelTrick3DScene`

Environment work:

- Installed Microsoft Visual Studio 2022 Build Tools.
- Installed C++ workload and verified `cl.exe`.
- Installed Manim successfully after Build Tools installation.
- Removed `MathTex` dependency from Manim scenes so the project does not require a full LaTeX install.

Rendered outputs:

- `outputs/phase1_LinearSVMMarginScene.mp4`
- `outputs/phase1_KernelTrick3DScene.mp4`

## 3. Phase 2 - Scikit-Learn SVM Decision Surface

Implemented:

- `src/data.py`
- `src/svm_engine.py`
- `src/metrics.py`
- `src/plotting.py`
- `scripts/export_phase2.py`

Key behavior:

- Uses real `sklearn.svm.SVC`.
- Computes true decision surface with `model.decision_function`.
- Extracts support vectors.
- Exports 2D and 3D Plotly HTML artifacts.
- Correctly remaps `make_circles` labels so `y=0` is blue core and `y=1` is red outer ring.

Outputs:

- `outputs/phase2_decision_boundary.html`
- `outputs/phase2_kernel_lift_3d.html`
- `outputs/phase2_metrics.json`

Metrics snapshot:

- Accuracy: 0.975
- Precision: 0.9672
- Recall: 0.9833
- F1: 0.9752
- AUC: 0.9986
- Support vectors: 40
- Label mapping OK: true
- RBF lift order OK: true

## 4. Phase 3 - Streamlit / Plotly Dashboard

Implemented:

- `app/streamlit_app.py`
- `.streamlit/config.toml`
- `src/theme.py`

Features:

- Dataset selector
- Kernel selector
- Controls for `C`, `gamma`, `degree`, noise, samples, mesh resolution, random seed
- 2D decision contour
- 3D kernel lift visualization
- Support vector highlighting
- Confusion matrix
- ROC curve and AUC
- Concept notes

Status:

- Streamlit server reachable at `http://localhost:8501`.

## 5. Image / Infographic Work

Edited and corrected:

- `image.jpg`
- `image.webp`
- `image_hyperplane_red_above_clean_manual_final.png`

Corrections:

- Removed hand-drawn red circles.
- Fixed red sample positions in 3D hyperplane diagrams.
- Ensured red points do not appear below the hyperplane.
- Made red point distribution non-Gaussian and irregular.
- Preserved the 2D diagrams and core infographic layout.

## 6. OpenCode / DeepSeek Setup

Worked on:

- `../opencode.json`
- `setup_opencode_deepseek.ps1`
- `OPENCODE_FIX.md`

Findings:

- `Unauthorized` was caused by a missing API key.
- DeepSeek API key was configured in Windows User environment.
- Authentication then passed, but DeepSeek returned `Insufficient Balance`.

Current blocker:

- DeepSeek account needs available balance/quota for OpenCode cloud model calls.

## 7. Validation

Commands:

```powershell
.\run_all_phases.ps1
.\validate_hw8.ps1
```

Final status:

- Phase 1: rendered successfully
- Phase 2: completed
- Phase 3: completed and reachable
- `compileall`: passed
- `pytest`: 6 passed

Main report:

- `outputs/phase_completion_report.md`
