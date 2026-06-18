"""Plotly figure builders for SVM visualizations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.metrics import MetricBundle
from src.svm_engine import DecisionMesh, TrainedSVM, kernel_lift_z, lifted_surface
from src.theme import COLORS, PLOTLY_LAYOUT


def decision_boundary_figure(
    X: np.ndarray,
    y: np.ndarray,
    trained: TrainedSVM,
    mesh: DecisionMesh,
) -> go.Figure:
    """Build a 2D contour figure with decision boundary, margins, and support vectors."""
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=mesh.xx[0],
            y=mesh.yy[:, 0],
            z=mesh.zz,
            colorscale=[
                [0.0, COLORS["class_outer"]],
                [0.5, "#111827"],
                [1.0, COLORS["class_inner"]],
            ],
            opacity=0.58,
            contours={"showlines": False},
            colorbar={"title": "f(x)", "thickness": 12},
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>score=%{z:.3f}<extra></extra>",
            name="Decision score",
        )
    )

    for level, dash, width, color, name in [
        (0, "solid", 4, COLORS["boundary"], "decision boundary"),
        (-1, "dash", 2, COLORS["margin"], "margin -1"),
        (1, "dash", 2, COLORS["margin"], "margin +1"),
    ]:
        fig.add_trace(
            go.Contour(
                x=mesh.xx[0],
                y=mesh.yy[:, 0],
                z=mesh.zz,
                showscale=False,
                contours={
                    "type": "constraint",
                    "operation": "=",
                    "value": level,
                    "showlabels": False,
                },
                line={"color": color, "width": width, "dash": dash},
                hoverinfo="skip",
                name=name,
            )
        )

    class_colors = np.where(y == 0, COLORS["class_inner"], COLORS["class_outer"])
    class_names = np.where(y == 0, "Blue core", "Red ring")
    scores = trained.model.decision_function(X)
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker={
                "size": 9,
                "color": class_colors,
                "line": {"width": 1, "color": "rgba(255,255,255,0.65)"},
                "opacity": 0.92,
            },
            customdata=np.c_[class_names, scores],
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>class=%{customdata[0]}<br>score=%{customdata[1]:.3f}<extra></extra>",
            name="samples",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trained.support_vectors[:, 0],
            y=trained.support_vectors[:, 1],
            mode="markers",
            marker={
                "size": 16,
                "symbol": "circle-open",
                "color": COLORS["amber"],
                "line": {"width": 3, "color": COLORS["amber"]},
            },
            hovertemplate="support vector<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            name="support vectors",
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="2D SVM Decision Space",
        xaxis_title="x",
        yaxis_title="y",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def kernel_lift_figure(
    X: np.ndarray,
    y: np.ndarray,
    trained: TrainedSVM,
    mesh: DecisionMesh,
) -> go.Figure:
    """Build a 3D educational kernel-lift figure."""
    z = kernel_lift_z(X, kernel=trained.kernel, gamma=float(trained.gamma), degree=trained.degree)
    surface_z = lifted_surface(mesh, kernel=trained.kernel, gamma=float(trained.gamma), degree=trained.degree)
    class_colors = np.where(y == 0, COLORS["class_inner"], COLORS["class_outer"])

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=mesh.xx,
            y=mesh.yy,
            z=surface_z,
            surfacecolor=mesh.zz,
            colorscale=[
                [0.0, COLORS["class_outer"]],
                [0.5, "#111827"],
                [1.0, COLORS["class_inner"]],
            ],
            opacity=0.38,
            showscale=False,
            hoverinfo="skip",
            name="kernel lift surface",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=z,
            mode="markers",
            marker={"size": 5, "color": class_colors, "opacity": 0.92},
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            name="lifted samples",
        )
    )

    boundary_mask = np.abs(mesh.zz) < np.percentile(np.abs(mesh.zz), 6)
    if np.any(boundary_mask):
        fig.add_trace(
            go.Scatter3d(
                x=mesh.xx[boundary_mask],
                y=mesh.yy[boundary_mask],
                z=surface_z[boundary_mask],
                mode="markers",
                marker={"size": 2, "color": COLORS["boundary"], "opacity": 0.42},
                hoverinfo="skip",
                name="decision trace",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="3D Kernel Lift Projection",
        scene={
            "xaxis": {"title": "x", "backgroundcolor": "#0B1120", "gridcolor": "#273449"},
            "yaxis": {"title": "y", "backgroundcolor": "#0B1120", "gridcolor": "#273449"},
            "zaxis": {"title": "kernel lift z", "backgroundcolor": "#0B1120", "gridcolor": "#273449"},
            "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.95}},
        },
        height=720,
    )
    return fig


def confusion_matrix_figure(metrics: MetricBundle) -> go.Figure:
    """Build a confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=metrics.confusion,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            colorscale=[[0, "#111827"], [1, COLORS["accent"]]],
            text=metrics.confusion,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>count=%{z}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Confusion Matrix", height=360)
    return fig


def roc_curve_figure(metrics: MetricBundle) -> go.Figure:
    """Build an ROC curve chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics.fpr,
            y=metrics.tpr,
            mode="lines",
            line={"color": COLORS["cyan"], "width": 4},
            name=f"AUC = {metrics.auc_score:.3f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": COLORS["muted"], "dash": "dash"},
            name="baseline",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=360,
    )
    return fig
