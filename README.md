# SVM Kernel Trick 3D Interactive Demo

HW8 is a complete Support Vector Machine teaching project. It shows how nonlinear 2D data, such as a blue center cluster surrounded by red outer samples, can be handled with SVM kernels and visualized through 2D/3D decision-space diagrams.

The Streamlit app is organized in a multi-page teaching style inspired by `ChenYuHsu413/L13-SVM`:

- Landing page
- SVM Concept
- Margin and Support Vectors
- Interactive SVM
- Kernel Trick
- Quiz

The project includes three phases:

1. **Phase 1 - Manim concept animation**
   Explains maximum margin, support vectors, and the kernel trick.
2. **Phase 2 - Scikit-Learn SVM engine**
   Trains a real `sklearn.svm.SVC` model and computes the true decision surface with `decision_function`.
3. **Phase 3 - Streamlit/Plotly dashboard**
   Provides an interactive dark glassmorphism UI for tuning dataset, kernel, `C`, `gamma`, and `degree`.

## Installation

Install the dashboard and core ML dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install Manim separately:

```powershell
python -m pip install -r requirements-manim.txt
```

Manim is best installed with Python 3.10-3.12. On Python 3.14, some Manim dependencies such as `moderngl` and `glcontext` may require Microsoft C++ Build Tools.

## Run The Dashboard

```powershell
streamlit run app/streamlit_app.py
```

Or use the helper script:

```powershell
.\run_dashboard.ps1
```

Then open:

```text
http://localhost:8501
```

For Streamlit Cloud or simple deployment, use the root entrypoint:

```powershell
streamlit run streamlit_app.py
```

## Dashboard Features

- Dataset selection: Blue Core / Red Ring, Moons, Linear, XOR
- Kernel selection: RBF Gaussian, Linear, Polynomial, Sigmoid
- Hyperparameter controls: `C`, `gamma`, `degree`, mesh resolution, random seed
- 2D Plotly decision contour from true `SVC.decision_function`
- 3D Plotly kernel lift visualization
- Support vectors highlighted with amber outlines
- Model diagnostics: accuracy, precision, recall, F1, confusion matrix, ROC/AUC

## Render Manim Animations

Preview quality:

```powershell
manim -pql animations/svm_manim.py LinearSVMMarginScene
manim -pql animations/svm_manim.py KernelTrick3DScene
```

Root convenience entrypoint:

```powershell
manim -pql phase1_manim_kernel_trick.py LinearSVMMarginScene
manim -pql phase1_manim_kernel_trick.py KernelTrick3DScene
```

Higher quality:

```powershell
manim -pqh animations/svm_manim.py KernelTrick3DScene
```

## Project Structure

```text
HW8/
  app/
    streamlit_app.py
  pages/
    1_SVM_Concept.py
    2_Margin_and_Support_Vectors.py
    3_Interactive_SVM.py
    4_Kernel_Trick.py
    5_Quiz.py
  animations/
    svm_manim.py
  src/
    data.py
    metrics.py
    plotting.py
    svm_engine.py
    theme.py
  tests/
    test_svm_engine.py
  .streamlit/
    config.toml
  image_hyperplane_random_points_final.png
  requirements.txt
  requirements-manim.txt
  svm_project_prompt.yaml
```

## Important Math Note

The 3D lift shown in the dashboard and Manim scene is an educational visualization:

```text
z = exp(-gamma * (x^2 + y^2))
```

The true RBF SVM does not explicitly map data to a simple 3D surface. Its decision function is:

```text
f(x) = sum_i alpha_i y_i K(x_i, x) + b
```

with:

```text
K(x_i, x) = exp(-gamma * ||x_i - x||^2)
```

The 2D decision boundary in this project is computed from the trained Scikit-Learn model, not hand-drawn.

## Validation

Run:

```powershell
.\validate_hw8.ps1
```

Or manually:

```powershell
python -m compileall src app animations
python -m pytest -q
```

Current expected result:

```text
6 passed
```

## Run All Three Phases

Run the full HW8 phase workflow:

```powershell
.\run_all_phases.ps1
```

This generates:

```text
outputs/phase_completion_report.md
outputs/phase1_LinearSVMMarginScene.mp4
outputs/phase1_KernelTrick3DScene.mp4
outputs/phase2_decision_boundary.html
outputs/phase2_kernel_lift_3d.html
outputs/phase2_metrics.json
```

Phase 1 Manim rendering requires a working Manim installation. If Manim cannot be installed on Python 3.14 because of `moderngl` / `glcontext`, the script records that environment blocker and still completes Phase 2, Phase 3, and validation.

## OpenCode Note

If Antigravity/OpenCode reports `Unauthorized`, check [OPENCODE_FIX.md](OPENCODE_FIX.md). If it reports `Insufficient Balance`, the API key is valid but the DeepSeek account has no available quota.
