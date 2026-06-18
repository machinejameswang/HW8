"""Manim scenes for SVM margin and kernel-trick demonstrations."""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE_C,
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    OUT,
    RED_C,
    RIGHT,
    UP,
    Arrow,
    Circle,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    NumberPlane,
    ParametricFunction,
    Scene,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    VGroup,
    Write,
)

BACKGROUND = "#090B10"
CLASS_INNER = "#38BDF8"
CLASS_OUTER = "#F43F5E"
ACCENT = "#F59E0B"
BOUNDARY = "#F8FAFC"
MARGIN = "#A78BFA"
GRID = "#334155"


def circle_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic concentric-circle points for animation."""
    inner_angles = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    inner = np.c_[0.55 * np.cos(inner_angles), 0.55 * np.sin(inner_angles)]
    outer = np.c_[1.55 * np.cos(outer_angles), 1.55 * np.sin(outer_angles)]
    return inner, outer


class LinearSVMMarginScene(Scene):
    """Animate how SVM selects a maximum-margin linear separator."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        plane = NumberPlane(
            x_range=(-4, 4, 1),
            y_range=(-3, 3, 1),
            background_line_style={"stroke_color": GRID, "stroke_opacity": 0.42},
        )
        title = Text("SVM chooses the widest margin", font_size=34, color=BOUNDARY).to_edge(UP)
        self.play(FadeIn(plane), Write(title))

        left_points = [(-2.4, -0.9), (-2.1, 0.2), (-1.7, 1.0), (-1.2, -0.4), (-2.7, 0.8)]
        right_points = [(1.3, -0.9), (1.8, 0.1), (2.3, 0.9), (2.6, -0.25), (1.2, 0.95)]
        dots = VGroup(
            *[Dot(plane.c2p(x, y), color=CLASS_OUTER, radius=0.08) for x, y in left_points],
            *[Dot(plane.c2p(x, y), color=CLASS_INNER, radius=0.08) for x, y in right_points],
        )
        self.play(FadeIn(dots))

        candidate_lines = VGroup(
            DashedLine(plane.c2p(-0.9, -2.7), plane.c2p(0.75, 2.7), color="#64748B"),
            DashedLine(plane.c2p(-0.35, -2.7), plane.c2p(0.15, 2.7), color="#64748B"),
            DashedLine(plane.c2p(-1.35, -2.7), plane.c2p(1.15, 2.7), color="#64748B"),
        )
        self.play(FadeIn(candidate_lines))
        self.wait(0.8)
        self.play(FadeOut(candidate_lines))

        hyperplane = DashedLine(plane.c2p(0, -2.6), plane.c2p(0, 2.6), color=BOUNDARY, stroke_width=6)
        margin_l = DashedLine(plane.c2p(-0.9, -2.6), plane.c2p(-0.9, 2.6), color=MARGIN, stroke_width=3)
        margin_r = DashedLine(plane.c2p(0.9, -2.6), plane.c2p(0.9, 2.6), color=MARGIN, stroke_width=3)
        equation = Text("w^T x + b = 0", font_size=26, color=BOUNDARY).to_corner(UP + RIGHT)
        self.play(FadeIn(hyperplane), FadeIn(margin_l), FadeIn(margin_r), Write(equation))

        support_marks = VGroup(
            Circle(radius=0.18, color=ACCENT, stroke_width=4).move_to(plane.c2p(-1.2, -0.4)),
            Circle(radius=0.18, color=ACCENT, stroke_width=4).move_to(plane.c2p(1.2, 0.95)),
            Circle(radius=0.18, color=ACCENT, stroke_width=4).move_to(plane.c2p(1.3, -0.9)),
        )
        label = Text("Support vectors define the margin", font_size=25, color=ACCENT).next_to(title, DOWN)
        self.play(FadeIn(support_marks), Write(label))

        arrow = Arrow(plane.c2p(-0.9, -2.25), plane.c2p(0.9, -2.25), buff=0, color=ACCENT)
        margin_formula = Text("margin = 2 / ||w||", font_size=24, color=ACCENT).next_to(arrow, DOWN)
        self.play(FadeIn(arrow), Write(margin_formula))
        self.wait(2)


class KernelTrick3DScene(ThreeDScene):
    """Animate a Gaussian-style lift that makes circular data linearly separable."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        axes = ThreeDAxes(
            x_range=(-2.2, 2.2, 1),
            y_range=(-2.2, 2.2, 1),
            z_range=(0, 1.3, 0.5),
            x_length=6,
            y_length=6,
            z_length=3.2,
            axis_config={"color": GRID},
        )
        title = Text("Kernel trick: lift 2D circles into 3D", font_size=32, color=BOUNDARY).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(FadeIn(axes), Write(title))

        inner, outer = circle_dataset()
        inner_2d = VGroup(*[Dot(axes.c2p(x, y, 0), color=CLASS_INNER, radius=0.055) for x, y in inner])
        outer_2d = VGroup(*[Dot(axes.c2p(x, y, 0), color=CLASS_OUTER, radius=0.055) for x, y in outer])
        self.play(FadeIn(inner_2d), FadeIn(outer_2d))

        bad_line = ParametricFunction(
            lambda t: axes.c2p(t, 0.45 * t - 0.1, 0),
            t_range=(-2.0, 2.0),
            color="#94A3B8",
        )
        note = Text("No 2D line separates the core from the ring", font_size=23, color="#CBD5E1").to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(bad_line), Write(note))
        self.wait(0.8)
        self.play(FadeOut(bad_line), FadeOut(note))

        gamma = 0.9

        def lift(point: np.ndarray) -> np.ndarray:
            x, y = point
            z = np.exp(-gamma * (x * x + y * y))
            return axes.c2p(x, y, z)

        lifted_inner = VGroup(*[Dot(lift(p), color=CLASS_INNER, radius=0.055) for p in inner])
        lifted_outer = VGroup(*[Dot(lift(p), color=CLASS_OUTER, radius=0.055) for p in outer])
        formula = Text("z = exp(-gamma * (x^2 + y^2))", font_size=22, color=ACCENT).to_corner(LEFT + UP)
        self.add_fixed_in_frame_mobjects(formula)
        self.move_camera(phi=64 * DEGREES, theta=-42 * DEGREES, run_time=1.4)
        self.play(FadeOut(inner_2d), FadeOut(outer_2d), FadeIn(lifted_inner), FadeIn(lifted_outer), Write(formula))

        surface = Surface(
            lambda u, v: axes.c2p(u, v, np.exp(-gamma * (u * u + v * v))),
            u_range=(-2.0, 2.0),
            v_range=(-2.0, 2.0),
            resolution=(30, 30),
            fill_opacity=0.22,
            checkerboard_colors=["#1E293B", "#111827"],
            stroke_color="#475569",
            stroke_width=0.4,
        )
        self.play(FadeIn(surface))

        plane = Surface(
            lambda u, v: axes.c2p(u, v, 0.34),
            u_range=(-2.0, 2.0),
            v_range=(-2.0, 2.0),
            resolution=(2, 2),
            fill_opacity=0.36,
            checkerboard_colors=[MARGIN, MARGIN],
            stroke_color=BOUNDARY,
        )
        plane_label = Text("linear hyperplane in lifted space", font_size=22, color=BOUNDARY).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(plane_label)
        self.play(FadeIn(plane), Write(plane_label))
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(plane_label))

        circle_boundary = ParametricFunction(
            lambda t: axes.c2p(1.08 * np.cos(t), 1.08 * np.sin(t), 0.02),
            t_range=(0, 2 * np.pi),
            color=BOUNDARY,
            stroke_width=5,
        )
        top_note = Text("Projected back to 2D, the plane becomes a circular boundary", font_size=22, color=BOUNDARY).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(top_note)
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=1.5)
        self.play(FadeIn(circle_boundary), Write(top_note))
        self.wait(2)
