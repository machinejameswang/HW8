# HW8 - SVM Kernel Trick 3D Interactive Demo

This project is a three-phase teaching demo for Support Vector Machine concepts:

1. Manim concept animations for SVM margin and kernel trick intuition.
2. Scikit-Learn SVM training with true RBF decision function visualization.
3. Streamlit + Plotly interactive teaching website ready for Streamlit Community Cloud.

## Project Structure

```text
.
├── animations/                  # Manim scenes
├── app/streamlit_app.py          # Main full interactive app implementation
├── pages/                        # Streamlit multipage lessons
├── scripts/export_phase2.py      # Generates Phase 2 HTML/metrics outputs
├── src/                          # Dataset, SVM engine, metrics, plotting, theme
├── tests/                        # Validation tests
├── outputs/                      # Rendered videos, HTML plots, metrics
├── streamlit_app.py              # Streamlit Cloud entrypoint
├── phase1_manim_kernel_trick.py  # Manim compatibility entrypoint
├── requirements.txt              # Streamlit Cloud dependencies
└── requirements-manim.txt        # Local Manim-only dependency
```

## Run Locally

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Local URL:

```text
http://localhost:8501
```

## Run All Phases Locally

```powershell
.\run_all_phases.ps1
```

This renders Manim videos, exports Scikit-Learn decision-surface artifacts, and checks the Streamlit app.

## Phase 1 - Manim Animation

Generated outputs:

```text
outputs/phase1_LinearSVMMarginScene.mp4
outputs/phase1_KernelTrick3DScene.mp4
```

Manual render command:

```powershell
python -m pip install -r requirements-manim.txt
manim -qh animations/svm_manim.py LinearSVMMarginScene
manim -qh animations/svm_manim.py KernelTrick3DScene
```

## Phase 2 - Scikit-Learn True Decision Surface

```powershell
python scripts/export_phase2.py
```

Generated outputs:

```text
outputs/phase2_decision_boundary.html
outputs/phase2_kernel_lift_3d.html
outputs/phase2_metrics.json
```

The SVM engine uses `sklearn.svm.SVC`, `decision_function`, and support vectors from the fitted model.

## Phase 3 - Streamlit / Plotly Teaching Site

The Streamlit app includes:

- concept pages with Manim video playback
- margin and support-vector lesson
- interactive SVM controls for `kernel`, `C`, `gamma`, `degree`, sample count, noise, and seed
- 2D decision boundary visualization
- 3D kernel-lift visualization
- short learning quiz

## Deploy To Streamlit Community Cloud

The repository is ready for deployment. Use these settings in Streamlit Cloud:

```text
Repository: machinejameswang/HW8
Branch: main
Main file path: streamlit_app.py
Python version: 3.11 recommended
```

Steps:

1. Go to <https://share.streamlit.io/>.
2. Sign in with the GitHub account that can access `machinejameswang/HW8`.
3. Click `Create app` or `New app`.
4. Choose `Deploy a public app from GitHub`.
5. Fill in the repository, branch, and main file path shown above.
6. Click `Deploy`.

No API keys or secrets are required for this HW8 app.

## Validation

```powershell
python -m pytest -q
```

Expected result:

```text
6 passed
```

## Notes

Manim is intentionally excluded from `requirements.txt` because Streamlit Cloud only needs to serve the generated videos and interactive app. Local Manim rendering uses `requirements-manim.txt`.
