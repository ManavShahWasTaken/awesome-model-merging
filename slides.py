"""
Model Merging — Manim Slides Presentation
==========================================
Render:  manim-slides render -qh slides.py HookSlide
         manim-slides render -qh slides.py LossLandscapeSlide
Present: manim-slides present HookSlide LossLandscapeSlide
Export:  manim-slides convert HookSlide presentation.html --one-file
"""

from manim import *
from manim_slides import Slide, ThreeDSlide
import numpy as np

# Speed up all animations globally
config.animation_run_time = 0.4


# ─── Color palette ───────────────────────────────────────────────────────────
SAFETY_COLOR = "#4FC3F7"   # light blue
CODE_COLOR = "#81C784"     # green
RLVR_COLOR = "#FFB74D"     # orange
MERGE_COLOR = "#CE93D8"    # purple
BG_GRAY = "#1e1e1e"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def model_box(label: str, color: str, width=2.2, height=0.8) -> VGroup:
    """A rounded rectangle with a centered label — represents a model."""
    rect = RoundedRectangle(
        corner_radius=0.15, width=width, height=height,
        stroke_color=color, fill_color=color, fill_opacity=0.15,
        stroke_width=2,
    )
    text = Text(label, font_size=20, color=color)
    text.move_to(rect.get_center())
    return VGroup(rect, text)


# ─── Slide 1: The Hook ──────────────────────────────────────────────────────

class HookSlide(Slide):
    def construct(self):
        # ── Beat 1: Title card ──
        title = Text("Model Merging", font_size=60, weight=BOLD)
        subtitle = Text(
            "Combining neural networks without retraining",
            font_size=24, color=GRAY_B,
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.next_slide()

        # ── Beat 2: The scenario — 3 teams, same base model ──
        self.play(FadeOut(title), FadeOut(subtitle))

        scenario_title = Text(
            "A common scenario at frontier labs",
            font_size=32, weight=BOLD,
        ).to_edge(UP, buff=0.6)

        # Base model at center-top
        base = model_box("Base Model", WHITE, width=2.4)
        base.shift(UP * 1.5)

        # Three specialist models branching out below
        safety = model_box("Safety &\nAlignment", SAFETY_COLOR)
        code = model_box("Code\nReasoning", CODE_COLOR)
        rlvr = model_box("RLVR /\nReasoning", RLVR_COLOR)

        specialists = VGroup(safety, code, rlvr)
        specialists.arrange(RIGHT, buff=1.0)
        specialists.shift(DOWN * 0.5)

        # Arrows from base to each specialist
        arrows_down = VGroup(*[
            Arrow(
                base.get_bottom(), spec.get_top(),
                buff=0.15, stroke_width=2, color=GRAY_B,
            )
            for spec in [safety, code, rlvr]
        ])

        # Team labels under each box
        team_labels = VGroup(*[
            Text(f"Team {i+1}", font_size=16, color=GRAY_C).next_to(spec, DOWN, buff=0.2)
            for i, spec in enumerate([safety, code, rlvr])
        ])

        self.play(Write(scenario_title))
        self.play(FadeIn(base, shift=DOWN * 0.3))
        self.play(
            *[GrowArrow(a) for a in arrows_down],
            *[FadeIn(s, shift=DOWN * 0.3) for s in specialists],
            *[FadeIn(t) for t in team_labels],
            run_time=1.5,
        )
        self.next_slide()

        # ── Beat 3: The problem — leadership wants ONE model ──
        problem = Text(
            '"We need a single model with all three capabilities."',
            font_size=24, color=YELLOW, slant=ITALIC,
        ).shift(DOWN * 2.0)

        self.play(FadeIn(problem, shift=UP * 0.2))
        self.next_slide()

        # ── Beat 4: Naive solution — retrain (expensive) ──
        # Fade out diagram to make room
        diagram_parts = VGroup(
            scenario_title, base, arrows_down, specialists, team_labels,
        )
        self.play(FadeOut(problem), FadeOut(diagram_parts))

        naive = VGroup(
            Text("Option A: Retrain from scratch", font_size=28, weight=BOLD),
            Text("• Need everyone's data and pipelines", font_size=22, color=GRAY_B),
            Text("• Expensive (months of GPU time)", font_size=22, color=GRAY_B),
            Text("• Risk regression on working capabilities", font_size=22, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        cross = Cross(stroke_color=RED, stroke_width=6)
        cross.scale(0.4).next_to(naive[0], RIGHT, buff=0.3)

        self.play(FadeIn(naive, shift=UP * 0.2))
        self.play(FadeIn(cross))
        self.next_slide()

        # ── Beat 5: The wild idea — just average the weights ──
        # Bring diagram back, clear Option A
        self.play(FadeOut(naive), FadeOut(cross))
        self.play(FadeIn(diagram_parts))

        wild_idea = Text(
            "Option B: Average the weights?",
            font_size=28, weight=BOLD, color=MERGE_COLOR,
        ).shift(DOWN * 1.8)

        # Converging arrows from specialists to a merged model
        merged = model_box("Merged Model", MERGE_COLOR, width=2.4)
        merged.shift(DOWN * 2.8)

        arrows_merge = VGroup(*[
            Arrow(
                spec.get_bottom(), merged.get_top(),
                buff=0.15, stroke_width=2, color=MERGE_COLOR,
            )
            for spec in [safety, code, rlvr]
        ])

        self.play(FadeIn(wild_idea, shift=UP * 0.2))
        self.next_slide()

        # Move text above the diagram as arrows grow (so they don't overlap)
        self.play(
            wild_idea.animate.next_to(scenario_title, DOWN, buff=0.3),
            *[GrowArrow(a) for a in arrows_merge],
            FadeIn(merged, shift=UP * 0.3),
            run_time=1.5,
        )
        self.next_slide()

        # ── Beat 6: The reveal — it works! ──
        # Clear everything for a clean reveal
        self.play(
            FadeOut(wild_idea), FadeOut(merged),
            FadeOut(arrows_merge), FadeOut(diagram_parts),
        )

        reveal = Text("It works.", font_size=48, weight=BOLD, color=GREEN)
        reveal.shift(UP * 0.5)

        formula = MathTex(
            r"\theta_{\text{merged}} = \frac{1}{N} \sum_{i=1}^{N} \theta_i",
            font_size=42,
        ).next_to(reveal, DOWN, buff=0.6)

        caveat = Text(
            "...with some caveats.",
            font_size=22, color=GRAY_B,
        ).next_to(formula, DOWN, buff=0.5)

        self.play(FadeIn(reveal, scale=1.3), run_time=0.8)
        self.next_slide()

        self.play(Write(formula), run_time=1.5)
        self.play(FadeIn(caveat, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 2: The Geometry of the Loss Landscape (2.1–2.3) ──────────────────

class LossLandscapeSlide(Slide):
    """2.1 (loss landscape + averaging), 2.2 (modes), 2.3 (curved connectivity).
    Opens with a complex heatmap, transitions to a clean 2D cross-section."""

    def loss_fn(self, x):
        """1D cross-section: two valleys with a ridge in the middle."""
        return 1.2 * np.exp(-2 * (x + 2)**2) + 1.2 * np.exp(-2 * (x - 2)**2)

    def loss_2d(self, x, y):
        """2D loss landscape for the heatmap — many basins, ridges, noise."""
        basins = (
            2.0 * np.exp(-((x + 2)**2 + (y - 1)**2) / 1.2)
            + 2.0 * np.exp(-((x - 2)**2 + (y + 0.5)**2) / 1.0)
            + 1.0 * np.exp(-((x - 0.5)**2 + (y + 2.5)**2) / 0.8)
            + 0.8 * np.exp(-((x + 1)**2 + (y + 2)**2) / 0.6)
        )
        bowl = 0.08 * (x**2 + y**2)
        noise = 0.15 * np.sin(3 * x) * np.cos(4 * y)
        return bowl + 3.0 - basins + noise

    def _make_heatmap_image(self, res=300):
        """Generate a loss landscape heatmap as a numpy RGB array."""
        xs = np.linspace(-4.5, 4.5, res)
        ys = np.linspace(-4.5, 4.5, res)
        X, Y = np.meshgrid(xs, ys)
        Z = self.loss_2d(X, Y)

        # Normalize to 0–1
        Z = (Z - Z.min()) / (Z.max() - Z.min())

        # Colormap: blue (low) → teal → green → yellow → red (high)
        r = np.clip(Z * 3 - 1, 0, 1)
        g = np.clip(1 - np.abs(Z - 0.4) * 3, 0, 1)
        b = np.clip(1 - Z * 2.5, 0, 1)

        img = np.stack([r, g, b], axis=-1)
        return (img * 255).astype(np.uint8)

    def _sgd_trajectory(self, x0, target_x, steps=25):
        """Simulate noisy SGD from x0 toward target_x on the 1D loss curve."""
        rng = np.random.default_rng(42 if target_x < 0 else 123)
        xs = [x0]
        x = x0
        for _ in range(steps):
            # Gradient descent toward target + noise
            grad = -0.15 * (x - target_x) + rng.normal(0, 0.08)
            x = x + grad
            xs.append(x)
        return xs

    def construct(self):
        # ── Beat 1: Section title ────────────────────────────────────────────
        section_title = Text(
            "The Geometry of the Loss Landscape",
            font_size=36, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        self.play(Write(section_title))
        self.next_slide()

        # ── Beat 2: Complex heatmap (bird's-eye view) ───────────────────────
        heatmap_array = self._make_heatmap_image()
        heatmap = ImageMobject(heatmap_array)
        heatmap.set_height(5).shift(DOWN * 0.2)

        heatmap_caption = Text(
            "The loss landscape is a high-dimensional terrain.\n"
            "Each point is a set of weights. Color = loss.",
            font_size=18, color=GRAY_B, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.3)

        self.play(FadeIn(heatmap, scale=1.05), run_time=1.5)
        self.play(FadeIn(heatmap_caption, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: Draw cross-section line, then transition to 2D ───────────
        cross_line = DashedLine(
            heatmap.get_left(), heatmap.get_right(),
            color=WHITE, stroke_width=3, dash_length=0.15,
        )
        cross_label = Text("Cross-section →", font_size=16, color=WHITE)
        cross_label.next_to(cross_line, UP, buff=0.1).shift(RIGHT * 2)

        self.play(Create(cross_line), FadeIn(cross_label))
        self.next_slide()

        # Fade out heatmap, build the 2D cross-section
        self.play(FadeOut(heatmap), FadeOut(heatmap_caption), FadeOut(cross_line), FadeOut(cross_label))

        axes = Axes(
            x_range=[-4, 4, 1], y_range=[0, 2, 0.5],
            x_length=10, y_length=4,
            axis_config={"include_ticks": False, "stroke_width": 2},
            tips=False,
        ).shift(DOWN * 0.3)

        x_label = Text("Parameter space", font_size=18, color=GRAY_B)
        x_label.next_to(axes.x_axis, DOWN, buff=0.15)
        y_label = Text("Loss", font_size=18, color=GRAY_B).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.15)

        loss_curve = axes.plot(
            lambda x: 1.8 - self.loss_fn(x) + 0.15 * x**2 * 0.05,
            x_range=[-3.8, 3.8], color=WHITE, stroke_width=3,
        )
        fill = axes.get_area(loss_curve, x_range=[-3.8, 3.8], color=[BLUE_D, TEAL], opacity=0.3)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=1)
        self.play(Create(loss_curve), FadeIn(fill), run_time=1.5)
        self.next_slide()

        # ── Beat 4: SGD animation — two models diverge from the ridge ────────
        def y_of(x):
            return 1.8 - self.loss_fn(x) + 0.15 * x**2 * 0.05

        # Two dots start near the center ridge
        start_x = 0.1
        sgd_dot_a = Dot(axes.c2p(start_x, y_of(start_x)), color=SAFETY_COLOR, radius=0.1)
        sgd_dot_b = Dot(axes.c2p(-start_x, y_of(-start_x)), color=RLVR_COLOR, radius=0.1)

        sgd_label = Text(
            "Two training runs from similar initializations...",
            font_size=18, color=GRAY_B,
        ).to_edge(DOWN, buff=0.3)

        self.play(FadeIn(sgd_dot_a, scale=1.5), FadeIn(sgd_dot_b, scale=1.5), FadeIn(sgd_label))

        # Generate noisy SGD paths
        traj_a = self._sgd_trajectory(start_x, -2.0, steps=20)
        traj_b = self._sgd_trajectory(-start_x, 2.0, steps=20)

        # Trace paths for both dots
        path_a = TracedPath(sgd_dot_a.get_center, stroke_color=SAFETY_COLOR, stroke_width=2, stroke_opacity=0.6)
        path_b = TracedPath(sgd_dot_b.get_center, stroke_color=RLVR_COLOR, stroke_width=2, stroke_opacity=0.6)
        self.add(path_a, path_b)

        # Animate both dots stepping through their trajectories
        for xa, xb in zip(traj_a[1:], traj_b[1:]):
            self.play(
                sgd_dot_a.animate.move_to(axes.c2p(xa, y_of(xa))),
                sgd_dot_b.animate.move_to(axes.c2p(xb, y_of(xb))),
                run_time=0.15, rate_func=linear,
            )

        self.play(FadeOut(sgd_label))

        # Label final positions
        x_a, x_b = traj_a[-1], traj_b[-1]
        dot_a = sgd_dot_a
        dot_b = sgd_dot_b
        label_a = MathTex(r"\theta_A", font_size=30, color=SAFETY_COLOR)
        label_b = MathTex(r"\theta_B", font_size=30, color=RLVR_COLOR)
        label_a.next_to(dot_a, DOWN, buff=0.2)
        label_b.next_to(dot_b, DOWN, buff=0.2)

        arrived = Text(
            "...end up in different valleys.",
            font_size=18, color=GRAY_B,
        ).to_edge(DOWN, buff=0.3)

        self.play(FadeIn(label_a), FadeIn(label_b), FadeIn(arrived))
        self.next_slide()
        self.play(FadeOut(arrived), FadeOut(path_a), FadeOut(path_b))

        # Straight line between them — goes over the ridge
        straight_line = DashedLine(
            dot_a.get_center(), dot_b.get_center(),
            color=RED, stroke_width=3, dash_length=0.1,
        )

        # Midpoint marker on the ridge
        mid_x = 0.0
        mid_y = 1.8 - self.loss_fn(mid_x) + 0.15 * mid_x**2 * 0.05
        mid_dot = Dot(axes.c2p(mid_x, mid_y), color=RED, radius=0.1)
        mid_label = Text("Average → high loss!", font_size=20, color=RED, weight=BOLD)
        mid_label.next_to(mid_dot, UP, buff=0.25)

        # Arrow from the valley floor up to the ridge to show the loss spike
        valley_floor_y = y_of(-2.0)  # approximate valley floor
        ridge_arrow = Arrow(
            axes.c2p(0, valley_floor_y), axes.c2p(0, mid_y),
            color=RED, stroke_width=3, buff=0.05,
        )

        self.play(Create(straight_line), run_time=1)
        self.play(FadeIn(mid_dot), FadeIn(mid_label), GrowArrow(ridge_arrow))
        self.next_slide()

        # ── 2.2  Define "mode" ───────────────────────────────────────────────
        self.play(
            FadeOut(straight_line), FadeOut(mid_dot),
            FadeOut(mid_label), FadeOut(ridge_arrow),
        )

        # Highlight each valley with a translucent region
        valley_a_region = axes.get_area(
            loss_curve, x_range=[-3.2, -0.8],
            color=SAFETY_COLOR, opacity=0.2,
        )
        valley_b_region = axes.get_area(
            loss_curve, x_range=[0.8, 3.2],
            color=RLVR_COLOR, opacity=0.2,
        )
        mode_a_text = Text("Mode A", font_size=22, color=SAFETY_COLOR, weight=BOLD)
        mode_b_text = Text("Mode B", font_size=22, color=RLVR_COLOR, weight=BOLD)
        mode_a_text.move_to(axes.c2p(-2, 1.2))
        mode_b_text.move_to(axes.c2p(2, 1.2))

        definition = Text(
            'A "mode" is a valley — a region of parameter space\n'
            'that a training run converges to.',
            font_size=20, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.4)

        self.play(
            FadeIn(valley_a_region), FadeIn(valley_b_region),
            FadeIn(mode_a_text), FadeIn(mode_b_text),
        )
        self.play(FadeIn(definition, shift=UP * 0.2))
        self.next_slide()

        # ── 2.3  Curved connectivity (paper figures) ─────────────────────────
        # Fade out everything from the 1D cross-section
        cross_section_parts = VGroup(
            section_title, axes, x_label, y_label, loss_curve, fill,
            dot_a, dot_b, label_a, label_b,
            valley_a_region, valley_b_region,
            mode_a_text, mode_b_text, definition,
        )
        self.play(FadeOut(cross_section_parts))

        conn_title = Text(
            "Curved Connectivity", font_size=34, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        # Garipov et al. — wide figure, show at top
        garipov_img = ImageMobject("assets/garipov et al.png")
        garipov_img.set_width(10).shift(UP * 0.3)
        garipov_caption = Text(
            '"Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs"'
            "  —  Garipov et al. (2018)",
            font_size=14, color=GRAY_B,
        ).next_to(garipov_img, DOWN, buff=0.15)

        self.play(FadeIn(conn_title))
        self.play(FadeIn(garipov_img), FadeIn(garipov_caption), run_time=1)
        self.next_slide()

        # Transition to Draxler et al.
        self.play(FadeOut(garipov_img), FadeOut(garipov_caption))

        draxler_img = ImageMobject("assets/draxler et al.png")
        draxler_img.set_height(4.2).shift(DOWN * 0.1)
        draxler_caption = Text(
            '"Essentially No Barriers in Neural Network Energy Landscape"'
            "  —  Draxler et al. (2018)",
            font_size=14, color=GRAY_B,
        ).next_to(draxler_img, DOWN, buff=0.15)

        self.play(FadeIn(draxler_img), FadeIn(draxler_caption), run_time=1)
        self.next_slide()

        # Punchline
        self.play(FadeOut(draxler_img), FadeOut(draxler_caption))

        punchline_lines = VGroup(
            Text(
                "Minima are connected by curves — not straight lines.",
                font_size=24,
            ),
            Text(
                "But weight averaging walks a straight line.",
                font_size=24, color=RED,
            ),
            Text(
                "Curved connectivity alone doesn't explain why merging works.",
                font_size=20, color=GRAY_B,
            ),
        ).arrange(DOWN, buff=0.35)

        self.play(FadeIn(punchline_lines[0], shift=UP * 0.2))
        self.play(FadeIn(punchline_lines[1], shift=UP * 0.2))
        self.play(FadeIn(punchline_lines[2], shift=UP * 0.2))
        self.next_slide()


# ─── Slide 2.4: Linear Mode Connectivity (Frankle et al., 2020) ──────────────
# (moved here from end of file to maintain presentation order)

class LMCSlide(ThreeDSlide):
    """3D visualization of the forking experiment and critical k."""

    BASIN_LEFT = np.array([-1.8, 0.0])
    BASIN_RIGHT = np.array([1.8, 0.0])

    def loss(self, x, y):
        """Two-basin surface with a separating ridge."""
        left = 2.3 * np.exp(-((x + 1.8) ** 2 + y**2) / 0.9)
        right = 2.3 * np.exp(-((x - 1.8) ** 2 + y**2) / 0.9)
        bowl = 0.10 * (x**2 + y**2)
        return bowl + 2.7 - left - right

    def quad_curve(self, p0, p1, p2, n=60):
        """Quadratic curve points in (x, y)."""
        return [
            (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            for t in np.linspace(0, 1, n)
        ]

    def surface_path(self, axes, xy_points, color=WHITE, width=4, z_offset=0.06):
        """Project xy points onto the loss surface."""
        pts = [axes.c2p(p[0], p[1], self.loss(p[0], p[1]) + z_offset) for p in xy_points]
        path = VMobject(color=color, stroke_width=width)
        path.set_points_smoothly(pts)
        return path

    def construct(self):
        self.set_camera_orientation(phi=50 * DEGREES, theta=-68 * DEGREES, zoom=0.68)

        scene_offset = DOWN * 2.0

        axes = ThreeDAxes(
            x_range=[-3.4, 3.4], y_range=[-3.4, 3.4], z_range=[-0.2, 3.6],
            x_length=7.2, y_length=7.2, z_length=3.6, tips=False,
        ).set_opacity(0).shift(scene_offset)

        surface = Surface(
            lambda u, v: axes.c2p(u, v, self.loss(u, v)),
            u_range=[-3.0, 3.0], v_range=[-3.0, 3.0],
            resolution=(44, 44),
            fill_opacity=0.55, stroke_width=0.2, stroke_opacity=0.2,
        )
        surface.set_fill_by_value(
            axes=axes, axis=2,
            colorscale=[
                (BLUE_D, 0.0), (TEAL, 0.6), (GREEN, 1.2),
                (YELLOW, 1.8), (ORANGE, 2.3), (RED, 3.2),
            ],
        )

        title = Text("From Curves to Straight Lines: When Does LMC Hold?", font_size=30, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        cite = Text("Frankle et al., 2020", font_size=18, color=GRAY_B)
        cite.next_to(title, DOWN, buff=0.12)
        self.add_fixed_in_frame_mobjects(title, cite)

        basin_l = Dot3D(axes.c2p(*self.BASIN_LEFT, self.loss(*self.BASIN_LEFT) + 0.08), color=SAFETY_COLOR, radius=0.08)
        basin_r = Dot3D(axes.c2p(*self.BASIN_RIGHT, self.loss(*self.BASIN_RIGHT) + 0.08), color=RLVR_COLOR, radius=0.08)
        basin_l_text = Text("Basin 1", font_size=18, color=SAFETY_COLOR)
        basin_r_text = Text("Basin 2", font_size=18, color=RLVR_COLOR)
        basin_l_text.next_to(basin_l, DOWN + LEFT, buff=0.18)
        basin_r_text.next_to(basin_r, DOWN + RIGHT, buff=0.18)
        self.add_fixed_orientation_mobjects(basin_l_text, basin_r_text)

        self.play(FadeIn(title), FadeIn(cite))
        self.play(Create(surface), run_time=2.0)
        self.play(FadeIn(basin_l), FadeIn(basin_r), FadeIn(basin_l_text), FadeIn(basin_r_text))
        self.next_slide()

        # Scenario A: fork early -> diverge to different basins
        scenario_a = Text("Scenario A: Fork too early", font_size=22, color=RED_B, weight=BOLD)
        scenario_a.to_corner(UL, buff=0.4).shift(DOWN * 0.7)
        self.add_fixed_in_frame_mobjects(scenario_a)

        p_start = np.array([0.0, 2.25])
        p_fork_early = np.array([0.0, 0.95])
        shared_early_xy = self.quad_curve(p_start, np.array([0.0, 1.7]), p_fork_early, n=40)
        shared_early = self.surface_path(axes, shared_early_xy, color=WHITE, width=4)

        fork_dot = Dot3D(axes.c2p(p_fork_early[0], p_fork_early[1], self.loss(*p_fork_early) + 0.10), color=YELLOW, radius=0.07)
        fork_k = MathTex("k", font_size=34, color=YELLOW)
        fork_k.next_to(fork_dot, UP, buff=0.14)
        self.add_fixed_orientation_mobjects(fork_k)

        branch_a_xy = self.quad_curve(p_fork_early, np.array([-1.0, 1.0]), self.BASIN_LEFT, n=48)
        branch_b_xy = self.quad_curve(p_fork_early, np.array([1.0, 1.0]), self.BASIN_RIGHT, n=48)
        branch_a = self.surface_path(axes, branch_a_xy, color=SAFETY_COLOR, width=4)
        branch_b = self.surface_path(axes, branch_b_xy, color=RLVR_COLOR, width=4)

        red_msg = Text("Different basins -> LMC fails", font_size=20, color=RED_B, weight=BOLD)
        red_msg.to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(red_msg)

        self.play(FadeIn(scenario_a))
        self.play(Create(shared_early), run_time=1.0)
        self.play(FadeIn(fork_dot), FadeIn(fork_k))
        self.play(Create(branch_a), Create(branch_b), run_time=1.6)
        self.play(FadeIn(red_msg))
        self.next_slide()

        # Scenario B: fork late (inside same basin) -> both branches stay there
        self.play(FadeOut(scenario_a), FadeOut(red_msg), FadeOut(shared_early), FadeOut(branch_a), FadeOut(branch_b), FadeOut(fork_dot), FadeOut(fork_k))

        scenario_b = Text("Scenario B: Fork after critical k", font_size=22, color=GREEN_B, weight=BOLD)
        scenario_b.to_corner(UL, buff=0.4).shift(DOWN * 0.7)
        self.add_fixed_in_frame_mobjects(scenario_b)

        p_fork_late = np.array([-1.45, 0.10])
        shared_late_xy = self.quad_curve(p_start, np.array([-0.8, 1.5]), p_fork_late, n=55)
        shared_late = self.surface_path(axes, shared_late_xy, color=WHITE, width=4)

        fork_late_dot = Dot3D(axes.c2p(p_fork_late[0], p_fork_late[1], self.loss(*p_fork_late) + 0.10), color=YELLOW, radius=0.07)
        fork_late_k = MathTex("k", font_size=34, color=YELLOW)
        fork_late_k.next_to(fork_late_dot, UP, buff=0.14)
        self.add_fixed_orientation_mobjects(fork_late_k)

        end_1 = np.array([-1.95, -0.05])
        end_2 = np.array([-1.60, -0.35])
        late_a_xy = self.quad_curve(p_fork_late, np.array([-1.75, -0.05]), end_1, n=34)
        late_b_xy = self.quad_curve(p_fork_late, np.array([-1.58, -0.22]), end_2, n=34)
        late_a = self.surface_path(axes, late_a_xy, color=SAFETY_COLOR, width=4)
        late_b = self.surface_path(axes, late_b_xy, color=RLVR_COLOR, width=4)

        green_msg = Text("Same basin -> LMC holds", font_size=20, color=GREEN_B, weight=BOLD)
        green_msg.to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(green_msg)

        self.play(FadeIn(scenario_b))
        self.play(Create(shared_late), run_time=1.2)
        self.play(FadeIn(fork_late_dot), FadeIn(fork_late_k))
        self.play(Create(late_a), Create(late_b), run_time=1.4)
        self.play(FadeIn(green_msg))
        self.next_slide()

        # Punchline — clear everything first, then show text cleanly
        self.play(*[FadeOut(m) for m in self.mobjects])

        punch_1 = Text(
            "Fine-tuning from a common pretrained\n"
            "base is a late fork.",
            font_size=28, weight=BOLD, line_spacing=1.4,
        )
        punch_2 = Text(
            "That is why linear interpolation\n"
            "often works in practice.",
            font_size=24, color=GRAY_B, line_spacing=1.4,
        )
        punch_3 = Text(
            "Qin et al. (2022) confirmed this for pre-trained\n"
            "language models — after just 5% of pre-training\n"
            "compute, fine-tuned models are linearly connected.",
            font_size=18, color=GRAY_C, line_spacing=1.3,
        )

        punch = VGroup(punch_1, punch_2, punch_3).arrange(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(punch)
        self.play(FadeIn(punch, shift=UP * 0.2), run_time=0.5)
        self.next_slide()


# ─── Slide 2.5: Permutation Symmetries, Git Re-Basin, REPAIR ─────────────────

class PermutationSlide(Slide):
    """2.5 — When LMC breaks: permutation symmetries, Git Re-Basin, REPAIR."""

    def _neuron_column(self, colors, x_center, y_top, spacing=0.6, radius=0.18):
        """Create a vertical column of colored neuron dots."""
        dots = VGroup()
        for i, c in enumerate(colors):
            d = Dot(
                point=[x_center, y_top - i * spacing, 0],
                color=c, radius=radius,
            )
            dots.add(d)
        return dots

    def _connect_layers(self, layer_a, layer_b, color=GRAY_D, width=1):
        """Draw lines from every neuron in layer_a to every neuron in layer_b."""
        lines = VGroup()
        for da in layer_a:
            for db in layer_b:
                lines.add(Line(
                    da.get_center(), db.get_center(),
                    stroke_color=color, stroke_width=width, stroke_opacity=0.35,
                ))
        return lines

    def construct(self):
        # ── Beat 1: Title + problem statement ────────────────────────────────
        title = Text(
            "When LMC Breaks: Permutation Symmetries",
            font_size=32, weight=BOLD,
        ).to_edge(UP, buff=0.4)
        cite = Text("Entezari et al. 2021", font_size=16, color=GRAY_B)
        cite.next_to(title, DOWN, buff=0.12)

        problem = Text(
            "LMC fails between independently trained models. Why?",
            font_size=22,
        ).next_to(cite, DOWN, buff=0.4)

        self.play(Write(title), FadeIn(cite))
        self.play(FadeIn(problem, shift=UP * 0.15))
        self.next_slide()

        # ── Beat 2: Neuron permutation diagram ──────────────────────────────
        self.play(FadeOut(problem))

        feat_colors = [RED, BLUE, GREEN, YELLOW]
        order_a = [RED, BLUE, GREEN, YELLOW]
        order_b = [GREEN, YELLOW, RED, BLUE]

        net_a_label = Text("Network A", font_size=18, weight=BOLD, color=SAFETY_COLOR)
        net_b_label = Text("Network B", font_size=18, weight=BOLD, color=RLVR_COLOR)

        x_a, x_b = -3.0, 3.0
        y_top = 1.0

        hidden_a = self._neuron_column(order_a, x_a, y_top)
        hidden_b = self._neuron_column(order_b, x_b, y_top)

        input_a = self._neuron_column([GRAY_C] * 2, x_a - 1.5, y_top - 0.3, spacing=0.6, radius=0.12)
        output_a = self._neuron_column([GRAY_C] * 2, x_a + 1.5, y_top - 0.3, spacing=0.6, radius=0.12)
        input_b = self._neuron_column([GRAY_C] * 2, x_b - 1.5, y_top - 0.3, spacing=0.6, radius=0.12)
        output_b = self._neuron_column([GRAY_C] * 2, x_b + 1.5, y_top - 0.3, spacing=0.6, radius=0.12)

        conn_a1 = self._connect_layers(input_a, hidden_a)
        conn_a2 = self._connect_layers(hidden_a, output_a)
        conn_b1 = self._connect_layers(input_b, hidden_b)
        conn_b2 = self._connect_layers(hidden_b, output_b)

        net_a_label.next_to(hidden_a, UP, buff=0.35)
        net_b_label.next_to(hidden_b, UP, buff=0.35)

        net_a = VGroup(input_a, hidden_a, output_a, conn_a1, conn_a2, net_a_label)
        net_b = VGroup(input_b, hidden_b, output_b, conn_b1, conn_b2, net_b_label)

        explanation = VGroup(
            Text("Same features learned — different positions.", font_size=20, weight=BOLD),
            Text("N neurons → N! equivalent arrangements.", font_size=18, color=GRAY_B),
            Text("Naive averaging mixes unrelated neurons.", font_size=18, color=RED),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.4)

        self.play(FadeIn(net_a), FadeIn(net_b), run_time=1.2)
        self.play(FadeIn(explanation, shift=UP * 0.15))
        self.next_slide()

        # ── Beat 3: Entezari's conjecture + Git Re-Basin ────────────────────
        self.play(FadeOut(net_a), FadeOut(net_b), FadeOut(explanation))

        # Text on the left — compact multi-line blocks
        conj_label = Text(
            "Entezari's conjecture (2021):",
            font_size=20, weight=BOLD,
        )
        conj_desc = Text(
            "Barriers aren't from fundamentally\n"
            "different solutions — they're from\n"
            "permutation misalignment.",
            font_size=16, color=GRAY_B, line_spacing=1.3,
        )
        rb_label = Text("Git Re-Basin", font_size=18, weight=BOLD, color=GREEN)
        rb_cite = Text("Ainsworth et al., ICLR 2023", font_size=14, color=GRAY_C)
        rb_desc = Text(
            "Found the aligning permutation.\n"
            "Zero-barrier LMC on CIFAR-10/100.",
            font_size=16, line_spacing=1.3,
        )
        rb_caveat = Text(
            "But alignment alone isn't enough.",
            font_size=18, weight=BOLD, color=RED,
        )

        conjecture_text = VGroup(
            conj_label, conj_desc, rb_label, rb_cite, rb_desc, rb_caveat,
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        conjecture_text.set_width(5.5)

        # Git Re-Basin figure on the right
        rebasin_img = ImageMobject("assets/git_rebasin.png")
        rebasin_img.set_height(3.5)
        rebasin_caption = Text(
            "Figure from Ainsworth et al., ICLR 2023",
            font_size=12, color=GRAY_C,
        )
        rebasin_group = Group(rebasin_img, rebasin_caption).arrange(DOWN, buff=0.1)

        # Lay out side by side
        beat3_layout = Group(conjecture_text, rebasin_group).arrange(RIGHT, buff=0.5)
        beat3_layout.next_to(cite, DOWN, buff=0.3)

        self.play(FadeIn(conjecture_text, shift=UP * 0.15), FadeIn(rebasin_group))
        self.next_slide()

        # ── Beat 4: What is variance collapse? ──────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        vc_title = Text(
            "Variance Collapse", font_size=30, weight=BOLD, color=RED,
        ).to_edge(UP, buff=0.5)
        vc_cite = Text(
            "REPAIR — Jordan et al., ICLR 2023", font_size=16, color=GRAY_B,
        ).next_to(vc_title, DOWN, buff=0.12)

        vc_problem = Text(
            "Even after perfect alignment, the midpoint model\n"
            "has < 1% accuracy on ImageNet. Why?",
            font_size=20, line_spacing=1.3,
        ).next_to(vc_cite, DOWN, buff=0.4)

        self.play(FadeIn(vc_title), FadeIn(vc_cite))
        self.play(FadeIn(vc_problem, shift=UP * 0.15))
        self.next_slide()

        # Explain what X is
        self.play(*[FadeOut(m) for m in self.mobjects])

        x_title = Text(
            "What is variance collapse?",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        x_explain = Text(
            "Pick one neuron — say, channel 45 of layer 8.\n"
            "Feed 5,000 images through. It produces 5,000 values.\n"
            "That distribution of values is X.",
            font_size=20, line_spacing=1.4,
        )

        x_defs = VGroup(
            VGroup(
                MathTex(r"X_1", font_size=28, color=SAFETY_COLOR),
                Text(" = that neuron's distribution in Network 1", font_size=18),
            ).arrange(RIGHT, buff=0.15),
            VGroup(
                MathTex(r"X_2", font_size=28, color=RLVR_COLOR),
                Text(" = same neuron (aligned) in Network 2", font_size=18),
            ).arrange(RIGHT, buff=0.15),
            VGroup(
                MathTex(r"X_\alpha", font_size=28, color=MERGE_COLOR),
                Text(" = same neuron in the merged network", font_size=18),
            ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        x_group = VGroup(x_explain, x_defs).arrange(
            DOWN, buff=0.4,
        ).next_to(x_title, DOWN, buff=0.4)

        self.play(Write(x_title))
        self.play(FadeIn(x_explain, shift=UP * 0.15))
        self.play(FadeIn(x_defs, shift=UP * 0.15))
        self.next_slide()

        # The variance formula + intuition
        self.play(*[FadeOut(m) for m in self.mobjects])

        var_title = Text(
            "The math of variance collapse",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        var_1 = Text(
            "At the midpoint (α = 0.5), even after alignment:",
            font_size=20,
        )
        var_formula = MathTex(
            r"\mathrm{Var}(X_\alpha) = \bigl(0.5 + 0.5 \cdot \mathrm{corr}(X_1, X_2)\bigr)"
            r"\cdot \mathrm{Var}(X_1)",
            font_size=30,
        )
        var_2 = Text(
            "Aligned neurons have corr ≈ 0.4 (similar, not identical).\n"
            "→ Variance drops to 70% per layer.",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )
        var_3 = Text(
            "Over 50 layers:  0.7⁵⁰ ≈ 0.00002",
            font_size=22, color=RED, weight=BOLD,
        )
        var_4 = Text(
            "The merged model's signal fades to nearly zero.",
            font_size=20, color=RED,
        )

        var_group = VGroup(var_1, var_formula, var_2, var_3, var_4).arrange(
            DOWN, buff=0.3,
        ).next_to(var_title, DOWN, buff=0.4)

        self.play(Write(var_title))
        self.play(FadeIn(var_1, shift=UP * 0.1))
        self.play(Write(var_formula))
        self.play(FadeIn(var_2, shift=UP * 0.1))
        self.play(FadeIn(var_3, shift=UP * 0.1))
        self.play(FadeIn(var_4, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: The REPAIR fix ──────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        fix_title = Text(
            "The fix: REPAIR",
            font_size=28, weight=BOLD, color=GREEN,
        ).to_edge(UP, buff=0.5)
        fix_cite = Text(
            "Jordan et al., ICLR 2023",
            font_size=16, color=GRAY_C,
        ).next_to(fix_title, DOWN, buff=0.15)

        fix_1 = Text(
            "1. Run a batch of data through the merged network.\n"
            "2. Measure actual variance at each layer.\n"
            "3. Rescale to match the originals.",
            font_size=20, line_spacing=1.4,
        )
        fix_2 = Text(
            "Simple. No retraining. Just statistics.",
            font_size=20, color=GRAY_B, slant=ITALIC,
        )
        fix_3 = Text(
            "ResNet-50 on ImageNet:  <1% → 56.5% accuracy\n"
            "ResNet-18 on CIFAR-10:  90% barrier reduction",
            font_size=20, color=GREEN, line_spacing=1.4,
        )

        fix_group = VGroup(fix_1, fix_2, fix_3).arrange(
            DOWN, buff=0.35,
        ).next_to(fix_cite, DOWN, buff=0.4)

        self.play(Write(fix_title), FadeIn(fix_cite))
        self.play(FadeIn(fix_1, shift=UP * 0.1))
        self.play(FadeIn(fix_2, shift=UP * 0.1))
        self.play(FadeIn(fix_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 6: Why this matters for us ──────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        tk_title = Text(
            "The practical takeaway",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        tk_1 = Text(
            "Independently trained models need:\n"
            "  1. Alignment (Git Re-Basin)\n"
            "  2. Variance repair (REPAIR)",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        tk_2 = Text(
            "But fine-tuned models from a shared base have:\n"
            "  • Already aligned neurons (same initialization)\n"
            "  • Highly correlated activations (corr ≈ 1, not 0.4)",
            font_size=21, color=GREEN, line_spacing=1.4,
        )
        tk_3 = Text(
            "Shared pretraining solves both problems at once.",
            font_size=24, weight=BOLD,
        )
        tk_4 = Text(
            "This is why every practical merging method\n"
            "assumes a common base.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        tk_group = VGroup(tk_1, tk_2, tk_3, tk_4).arrange(
            DOWN, buff=0.35,
        ).next_to(tk_title, DOWN, buff=0.4)

        self.play(Write(tk_title))
        self.play(FadeIn(tk_1, shift=UP * 0.1))
        self.play(FadeIn(tk_2, shift=UP * 0.1))
        self.play(FadeIn(tk_3, shift=UP * 0.1))
        self.play(FadeIn(tk_4, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 2.8: Cross-Task Linearity (Zhou et al., ICML 2024) ────────────────

class CTLSlide(Slide):
    """Cross-Task Linearity: the theoretical climax of Section 2."""

    def construct(self):
        # ── Beat 1: Title + the question ─────────────────────────────────────
        title = Text(
            "Cross-Task Linearity", font_size=36, weight=BOLD,
        ).to_edge(UP, buff=0.4)
        cite = Text(
            "Zhou et al., ICML 2024", font_size=16, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.12)

        question = VGroup(
            Text(
                "So far, LMC studied models forked onto\n"
                "the same task with different SGD noise.",
                font_size=22, line_spacing=1.4,
            ),
            Text(
                "What happens when models are fine-tuned\n"
                "on different tasks?",
                font_size=24, weight=BOLD, color=YELLOW, line_spacing=1.4,
            ),
        ).arrange(DOWN, buff=0.4).next_to(cite, DOWN, buff=0.5)

        self.play(Write(title), FadeIn(cite))
        self.play(FadeIn(question, shift=UP * 0.15))
        self.next_slide()

        # ── Beat 2: Hierarchy table ──────────────────────────────────────────
        self.play(FadeOut(question))

        # Build table as rows of VGroups for controlled animation
        header = VGroup(
            Text("Property", font_size=18, weight=BOLD, color=GRAY_C),
            Text("Paper", font_size=18, weight=BOLD, color=GRAY_C),
            Text("Scope", font_size=18, weight=BOLD, color=GRAY_C),
            Text("What it says", font_size=18, weight=BOLD, color=GRAY_C),
        ).arrange(RIGHT, buff=1.2)

        row_lmc = VGroup(
            Text("LMC", font_size=20, weight=BOLD),
            Text("Frankle 2020", font_size=18, color=GRAY_B),
            Text("Same task", font_size=18),
            Text("Loss stays flat", font_size=18),
        ).arrange(RIGHT, buff=1.2)

        row_llfc = VGroup(
            Text("LLFC", font_size=20, weight=BOLD),
            Text("Zhou 2023", font_size=18, color=GRAY_B),
            Text("Same task", font_size=18),
            Text("Features ∝ interpolation", font_size=18),
        ).arrange(RIGHT, buff=1.2)

        row_ctl = VGroup(
            Text("CTL", font_size=20, weight=BOLD, color=YELLOW),
            Text("Zhou 2024", font_size=18, color=YELLOW),
            Text("Different tasks", font_size=18, color=YELLOW, weight=BOLD),
            Text("Features ≈ interpolation", font_size=18, color=YELLOW),
        ).arrange(RIGHT, buff=1.2)

        table = VGroup(header, row_lmc, row_llfc, row_ctl)
        table.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        table.next_to(cite, DOWN, buff=0.4)

        # Align columns by centering each row relative to header
        for row in [row_lmc, row_llfc, row_ctl]:
            for i, cell in enumerate(row):
                cell.move_to([header[i].get_x(), cell.get_y(), 0])

        self.play(FadeIn(header))
        self.play(FadeIn(row_lmc, shift=UP * 0.1))
        self.play(FadeIn(row_llfc, shift=UP * 0.1))
        self.play(FadeIn(row_ctl, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: Cat example + conjecture ─────────────────────────────────
        self.play(FadeOut(VGroup(header, row_lmc, row_llfc, row_ctl)))

        cat_title = Text(
            "What does CTL actually say?", font_size=26, weight=BOLD,
        ).next_to(cite, DOWN, buff=0.35)

        cat_desc = Text(
            "Take one image. Feed it through three models:",
            font_size=20,
        )
        cat_models = Text(
            "Model A (task i)  →  features a at layer l\n"
            "Model B (task j)  →  features b at layer l\n"
            "Merged (0.5·θ_A + 0.5·θ_B)  →  features c at layer l",
            font_size=19, color=GRAY_B, line_spacing=1.4,
        )
        cat_ctl = VGroup(
            Text("CTL says:  ", font_size=22, weight=BOLD),
            MathTex(
                r"\mathbf{c} \approx 0.5\,\mathbf{a} + 0.5\,\mathbf{b}",
                font_size=28, color=YELLOW,
            ),
        ).arrange(RIGHT, buff=0.12)

        cat_lines = VGroup(cat_desc, cat_models, cat_ctl).arrange(
            DOWN, buff=0.35,
        )
        cat_lines.next_to(cat_title, DOWN, buff=0.3)

        self.play(FadeIn(cat_title))
        self.play(FadeIn(cat_lines, shift=UP * 0.15))
        self.next_slide()

        # Conjecture + conditions + scope
        self.play(FadeOut(cat_title), FadeOut(cat_lines))

        conj_1 = Text(
            "The conjecture:",
            font_size=22, weight=BOLD,
        )
        conj_2 = Text(
            "In the pretraining-finetuning paradigm,\n"
            "neural networks approximately function as\n"
            "linear maps from parameter space to feature space.",
            font_size=20, color=YELLOW, line_spacing=1.4,
        )
        conj_3 = Text(
            "Two conditions for CTL to emerge:\n"
            "  (a) Flat loss landscape around pretrained weights\n"
            "  (b) Fine-tuning moves weights only slightly",
            font_size=18, color=GRAY_B, line_spacing=1.4,
        )
        conj_4 = Text(
            "Tested on: Rotated MNIST (MLP), Split CIFAR-100\n"
            "(ResNet-18), ViTs, and T5 on text datasets.\n"
            "An interesting result — may not hold at frontier scale.",
            font_size=17, color=GRAY_C, line_spacing=1.3,
        )

        conjecture = VGroup(conj_1, conj_2, conj_3, conj_4).arrange(
            DOWN, buff=0.3,
        ).shift(UP * 0.3)

        self.play(FadeIn(conjecture, shift=UP * 0.15))
        self.next_slide()

        # ── Beat 4: Payoff formula + transition ──────────────────────────────
        self.play(FadeOut(title), FadeOut(cite), FadeOut(conjecture))

        payoff_formula = MathTex(
            r"f^{(\ell)}\!\bigl(\alpha\,\theta_i + (1{-}\alpha)\,\theta_j\bigr)"
            r"\;\approx\;"
            r"\alpha\, f^{(\ell)}(\theta_i) + (1{-}\alpha)\, f^{(\ell)}(\theta_j)",
            font_size=32,
        ).shift(UP * 1.0)

        payoff_lines = VGroup(
            Text(
                "If this holds: averaging weights ≈ averaging features.",
                font_size=22, weight=BOLD,
            ),
            Text(
                "Operations on parameters translate to operations on representations.",
                font_size=20, color=GRAY_B,
            ),
        ).arrange(DOWN, buff=0.2).next_to(payoff_formula, DOWN, buff=0.6)

        transition = Text(
            "The weight space around a pretrained model is approximately linear.\n"
            "Now let's talk about how to navigate it.",
            font_size=22, line_spacing=1.3, color=YELLOW,
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(payoff_formula), run_time=1.5)
        self.play(FadeIn(payoff_lines, shift=UP * 0.15))
        self.next_slide()

        self.play(FadeIn(transition, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 3: Task Arithmetic (§3) ──────────────────────────────────────────

class TaskArithmeticSlide(Slide):
    """Task vectors: definition, three operations, the reframing."""

    def construct(self):
        # Shared citation footer shown throughout
        cite = Text(
            "Ilharco et al., ICLR 2023",
            font_size=16, color=GRAY_C,
        ).to_corner(DR, buff=0.3)

        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Task Arithmetic", font_size=48, weight=BOLD)
        paper = Text("Ilharco et al., ICLR 2023", font_size=22, color=GRAY_B)
        paper.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(paper), FadeIn(cite))

        # ── Persistent vector diagram (right side) ───────────────────────────
        # A lightweight 2D plane representing "weight space"
        plane = NumberPlane(
            x_range=[-0.5, 4.5], y_range=[-2, 4],
            x_length=5.5, y_length=5,
            background_line_style={"stroke_opacity": 0.15},
            axis_config={"stroke_opacity": 0.3},
        ).shift(RIGHT * 2.8 + DOWN * 0.2)

        origin = plane.c2p(0, 0)
        origin_dot = Dot(origin, color=WHITE, radius=0.08)
        origin_label = MathTex(r"\theta_{\text{pre}}", font_size=26).next_to(
            origin_dot, DL, buff=0.1,
        )

        # Task vector endpoints in plane coordinates
        # (kept short so their sum stays within the visible frame)
        a_end = plane.c2p(2.5, 1.2)
        b_end = plane.c2p(1, 2.2)
        c_end = plane.c2p(3, 0.5)

        # ── Beat 2: Define task vectors ──────────────────────────────────────
        # Left-side text
        defn_title = Text("Task Vectors", font_size=28, weight=BOLD).to_edge(
            UP, buff=0.5,
        ).shift(LEFT * 3.2)

        defn_formula = MathTex(
            r"\tau_A = \theta_A - \theta_{\text{pre}}",
            font_size=34,
        ).next_to(defn_title, DOWN, buff=0.4, aligned_edge=LEFT)

        defn_text = Text(
            "Fine-tuning moves the model\n"
            "in some direction from the\n"
            "pretrained weights.\n\n"
            "That direction is the\n"
            "task vector.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        ).next_to(defn_formula, DOWN, buff=0.4, aligned_edge=LEFT)

        # Vector arrow for τ_A
        arrow_a = Arrow(origin, a_end, buff=0, stroke_width=4, color=SAFETY_COLOR)
        label_a = MathTex(r"\tau_A", font_size=28, color=SAFETY_COLOR)
        label_a.next_to(arrow_a.get_center(), UR, buff=0.1)

        dot_a = Dot(a_end, color=SAFETY_COLOR, radius=0.06)
        dot_a_label = MathTex(r"\theta_A", font_size=22, color=SAFETY_COLOR)
        dot_a_label.next_to(dot_a, RIGHT, buff=0.1)

        self.play(FadeIn(plane), FadeIn(origin_dot), Write(origin_label))
        self.play(Write(defn_title), Write(defn_formula))
        self.play(FadeIn(defn_text, shift=UP * 0.1))
        self.play(
            GrowArrow(arrow_a), FadeIn(label_a),
            FadeIn(dot_a), FadeIn(dot_a_label),
        )
        self.next_slide()

        # ── Beat 3: Multiple task vectors ────────────────────────────────────
        arrow_b = Arrow(origin, b_end, buff=0, stroke_width=4, color=CODE_COLOR)
        label_b = MathTex(r"\tau_B", font_size=28, color=CODE_COLOR)
        label_b.next_to(arrow_b.get_center(), LEFT, buff=0.1)

        arrow_c = Arrow(origin, c_end, buff=0, stroke_width=4, color=RLVR_COLOR)
        label_c = MathTex(r"\tau_C", font_size=28, color=RLVR_COLOR)
        label_c.next_to(arrow_c.get_center(), DR, buff=0.1)

        multi_text = Text(
            "Each task vector encodes\n"
            "everything the model learned\n"
            "about that task.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        ).move_to(defn_text, aligned_edge=LEFT)

        self.play(
            GrowArrow(arrow_b), FadeIn(label_b),
            GrowArrow(arrow_c), FadeIn(label_c),
            FadeTransform(defn_text, multi_text),
        )
        self.next_slide()

        # ── Beat 4: Addition ─────────────────────────────────────────────────
        # Remove τ_C, keep A and B for clarity
        self.play(
            FadeOut(arrow_c), FadeOut(label_c),
            FadeOut(defn_title), FadeOut(defn_formula), FadeOut(multi_text),
            FadeOut(dot_a), FadeOut(dot_a_label),
        )

        add_title = Text("Addition", font_size=28, weight=BOLD, color=GREEN).to_edge(
            UP, buff=0.5,
        ).shift(LEFT * 3.2)

        add_formula = MathTex(
            r"\theta_{\text{pre}} + \tau_A + \tau_B",
            font_size=34,
        ).next_to(add_title, DOWN, buff=0.4, aligned_edge=LEFT)

        add_text = Text(
            "A model that can do\n"
            "both tasks.\n\n"
            "This is what the Hook\n"
            "scenario was doing —\n"
            "just stated precisely.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        ).next_to(add_formula, DOWN, buff=0.4, aligned_edge=LEFT)

        # Vector addition: translate τ_B to tip of τ_A
        b_offset = np.array(a_end) - np.array(origin)
        sum_end = np.array(b_end) + b_offset
        arrow_b_shifted = Arrow(
            a_end, sum_end, buff=0,
            stroke_width=4, color=CODE_COLOR,
        )
        arrow_b_shifted_label = MathTex(
            r"\tau_B", font_size=28, color=CODE_COLOR,
        ).next_to(arrow_b_shifted.get_center(), LEFT, buff=0.1)

        # Dashed line showing the parallelogram
        dash_a = DashedLine(b_end, sum_end, color=GRAY_C, stroke_width=1.5)
        dash_b = DashedLine(a_end, sum_end, color=GRAY_C, stroke_width=1.5)

        merged_dot = Dot(sum_end, color=MERGE_COLOR, radius=0.1)
        merged_label = MathTex(
            r"\theta_{\text{merged}}", font_size=24, color=MERGE_COLOR,
        ).next_to(merged_dot, UR, buff=0.1)

        self.play(Write(add_title), Write(add_formula), FadeIn(add_text, shift=UP * 0.1))
        self.play(
            TransformFromCopy(arrow_b, arrow_b_shifted),
            FadeIn(arrow_b_shifted_label),
            Create(dash_a), Create(dash_b),
        )
        self.play(FadeIn(merged_dot, scale=1.5), Write(merged_label))
        self.next_slide()

        # ── Beat 5: Negation ─────────────────────────────────────────────────
        # Clear addition visuals
        add_visuals = VGroup(
            arrow_b_shifted, arrow_b_shifted_label,
            dash_a, dash_b, merged_dot, merged_label,
            arrow_b, label_b, add_title, add_formula, add_text,
        )
        self.play(FadeOut(add_visuals))

        neg_title = Text("Negation", font_size=28, weight=BOLD, color=RED).to_edge(
            UP, buff=0.5,
        ).shift(LEFT * 3.2)

        neg_formula = MathTex(
            r"\theta_{\text{pre}} - \tau_A",
            font_size=34,
        ).next_to(neg_title, DOWN, buff=0.4, aligned_edge=LEFT)

        neg_text = Text(
            "A model that forgets task A.\n\n"
            "Useful for unlearning —\n"
            "e.g. removing toxic behavior.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        ).next_to(neg_formula, DOWN, buff=0.4, aligned_edge=LEFT)

        # Flip τ_A into the opposite direction
        neg_end = 2 * np.array(origin) - np.array(a_end)
        arrow_neg = Arrow(
            origin, neg_end, buff=0, stroke_width=4, color=RED,
            stroke_opacity=0.8,
        )
        label_neg = MathTex(r"-\tau_A", font_size=28, color=RED)
        label_neg.next_to(arrow_neg.get_center(), DL, buff=0.1)

        forgot_dot = Dot(neg_end, color=RED, radius=0.08)
        forgot_label = MathTex(
            r"\theta_{\text{forgot}}", font_size=22, color=RED,
        ).next_to(forgot_dot, LEFT, buff=0.1)

        # Fade existing τ_A to dashed
        arrow_a_ghost = DashedLine(
            origin, a_end, color=SAFETY_COLOR, stroke_width=2, stroke_opacity=0.4,
        )

        self.play(Write(neg_title), Write(neg_formula), FadeIn(neg_text, shift=UP * 0.1))
        self.play(
            arrow_a.animate.set_opacity(0.25),
            label_a.animate.set_opacity(0.25),
            GrowArrow(arrow_neg), FadeIn(label_neg),
        )
        self.play(FadeIn(forgot_dot), Write(forgot_label))
        self.next_slide()

        # ── Beat 6: Scaling ──────────────────────────────────────────────────
        neg_visuals = VGroup(
            arrow_neg, label_neg, forgot_dot, forgot_label,
            neg_title, neg_formula, neg_text,
        )
        self.play(
            FadeOut(neg_visuals),
            arrow_a.animate.set_opacity(1),
            label_a.animate.set_opacity(1),
        )

        scale_title = Text(
            "Scaling", font_size=28, weight=BOLD, color=YELLOW,
        ).to_edge(UP, buff=0.5).shift(LEFT * 3.2)

        scale_formula = MathTex(
            r"\theta_{\text{pre}} + \lambda \, \tau_A",
            font_size=34,
        ).next_to(scale_title, DOWN, buff=0.4, aligned_edge=LEFT)

        scale_text = Text(
            "Control the strength of\n"
            "a capability.\n\n"
            "λ > 1  amplifies\n"
            "λ < 1  dampens",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        ).next_to(scale_formula, DOWN, buff=0.4, aligned_edge=LEFT)

        # Show three scaled versions of τ_A
        a_dir = (np.array(a_end) - np.array(origin))

        half_end = np.array(origin) + 0.5 * a_dir
        full_end = a_end
        big_end = np.array(origin) + 1.5 * a_dir

        arrow_half = Arrow(
            origin, half_end, buff=0, stroke_width=3,
            color=SAFETY_COLOR, stroke_opacity=0.5,
        )
        arrow_big = Arrow(
            origin, big_end, buff=0, stroke_width=5, color=YELLOW,
        )
        lbl_half = MathTex(r"\lambda=0.5", font_size=18, color=GRAY_B).next_to(
            half_end, RIGHT, buff=0.1,
        )
        lbl_full = MathTex(r"\lambda=1", font_size=18, color=SAFETY_COLOR).next_to(
            full_end, RIGHT, buff=0.1,
        )
        lbl_big = MathTex(r"\lambda=1.5", font_size=18, color=YELLOW).next_to(
            big_end, RIGHT, buff=0.1,
        )

        self.play(Write(scale_title), Write(scale_formula), FadeIn(scale_text, shift=UP * 0.1))
        self.play(GrowArrow(arrow_half), FadeIn(lbl_half))
        self.play(FadeIn(lbl_full))  # arrow_a already visible
        self.play(GrowArrow(arrow_big), FadeIn(lbl_big))
        self.next_slide()

        # ── Beat 7: The reframing ────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        old = Text(
            '"Averaging models"',
            font_size=32, color=GRAY_C,
        ).shift(UP * 0.8)
        # Strikethrough line
        strike = Line(
            old.get_left() + LEFT * 0.1,
            old.get_right() + RIGHT * 0.1,
            color=RED, stroke_width=3,
        )

        new = Text(
            '"Combining learned capabilities"',
            font_size=36, weight=BOLD, color=GREEN,
        ).next_to(old, DOWN, buff=0.6)

        bottom = Text(
            "Merging is vector arithmetic in a meaningful space.",
            font_size=22, color=GRAY_B,
        ).next_to(new, DOWN, buff=0.6)

        self.play(FadeIn(old))
        self.play(Create(strike))
        self.play(FadeIn(new, shift=UP * 0.2))
        self.play(FadeIn(bottom, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 4: The Interference Problem (§4) ─────────────────────────────────

class InterferenceSlide(Slide):
    """Why task vector addition isn't perfect: interference from overlap."""

    # Parameter values for the bar chart demo (12 params)
    # Orthogonal case: A and B touch completely different parameters
    ORTHO_A = np.array([0.6, 0.4, 0.8, 0.3, 0, 0, 0, 0, 0, 0, 0, 0])
    ORTHO_B = np.array([0, 0, 0, 0, 0, 0, 0.5, 0.7, 0.3, 0.6, 0.9, 0.4])

    # Overlap case: some shared parameters with conflicts
    OVER_A = np.array([0.6, 0.4, 0.8, 0.3, 0.5, -0.3, 0.2, 0, 0, 0, 0, 0])
    OVER_B = np.array([0, 0, 0, 0, -0.4, 0.6, -0.3, 0.5, 0.7, 0.3, 0.6, 0.4])

    def _bar_row(self, values, color, y_center, bar_width=0.3, max_h=1.2):
        """Build a row of vertical bars for a parameter vector."""
        bars = VGroup()
        n = len(values)
        total_w = n * bar_width * 1.4
        x_start = -total_w / 2
        for i, v in enumerate(values):
            x = x_start + i * bar_width * 1.4
            h = abs(v) * max_h
            if h < 0.02:
                # Near-zero: show a tiny stub
                bar = Line(
                    [x, y_center, 0], [x, y_center + 0.03, 0],
                    stroke_width=bar_width * 18, color=GRAY_D,
                )
            else:
                direction = 1 if v >= 0 else -1
                bar = Rectangle(
                    width=bar_width, height=h,
                    fill_color=color, fill_opacity=0.8,
                    stroke_width=0.5, stroke_color=WHITE,
                )
                bar.move_to([x, y_center + direction * h / 2, 0])
            bars.add(bar)
        return bars

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("The Interference Problem", font_size=48, weight=BOLD)
        subtitle = Text(
            '"Doesn\'t addition just work?"',
            font_size=24, color=GRAY_B, slant=ITALIC,
        ).next_to(title, DOWN, buff=0.3)

        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(subtitle))

        # ── Beat 2: The obvious question ─────────────────────────────────────
        q1 = Text(
            "Because of CTL, addition in weight space\n"
            "produces predictable results in feature space.",
            font_size=22, line_spacing=1.4,
        )
        q2 = Text(
            "But predictable ≠ desirable.",
            font_size=26, color=YELLOW, weight=BOLD,
        )
        q3 = Text(
            "Task vectors can have overlapping, conflicting,\n"
            "or redundant components.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        q4 = Text(
            "Linearity tells you what the merged features\n"
            "look like. Not that they're useful.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        q_group = VGroup(q1, q2, q3, q4).arrange(
            DOWN, buff=0.4, aligned_edge=LEFT,
        )

        self.play(FadeIn(q_group, shift=UP * 0.2))
        self.next_slide()
        self.play(FadeOut(q_group))

        # ── Beat 3: Parameter bars — orthogonal (happy path) ─────────────────
        ortho_title = Text(
            "Orthogonal task vectors — no interference",
            font_size=24, weight=BOLD, color=GREEN,
        ).to_edge(UP, buff=0.5)

        # Labels for rows
        lbl_a = MathTex(r"\tau_A", font_size=28, color=SAFETY_COLOR).shift(LEFT * 4.5 + UP * 1.3)
        lbl_b = MathTex(r"\tau_B", font_size=28, color=CODE_COLOR).shift(LEFT * 4.5 + DOWN * 0.1)
        lbl_sum = MathTex(r"\tau_A + \tau_B", font_size=24, color=WHITE).shift(LEFT * 4.5 + DOWN * 1.8)

        bars_oa = self._bar_row(self.ORTHO_A, SAFETY_COLOR, y_center=1.3)
        bars_ob = self._bar_row(self.ORTHO_B, CODE_COLOR, y_center=-0.1)
        bars_osum = self._bar_row(self.ORTHO_A + self.ORTHO_B, MERGE_COLOR, y_center=-1.8)

        # Separator lines
        sep1 = DashedLine(LEFT * 4, RIGHT * 4, color=GRAY_D, stroke_width=1).shift(DOWN * 0.8)
        plus_sign = MathTex("+", font_size=30).shift(LEFT * 4.5 + DOWN * 0.8)

        self.play(Write(ortho_title))
        self.play(FadeIn(lbl_a), FadeIn(bars_oa))
        self.play(FadeIn(lbl_b), FadeIn(bars_ob))
        self.play(FadeIn(sep1), Write(plus_sign))
        self.play(FadeIn(lbl_sum), FadeIn(bars_osum))
        self.next_slide()

        # ── Beat 4: Parameter bars — overlap (interference) ──────────────────
        ortho_all = VGroup(
            ortho_title, lbl_a, lbl_b, lbl_sum,
            bars_oa, bars_ob, bars_osum, sep1, plus_sign,
        )
        self.play(FadeOut(ortho_all))

        over_title = Text(
            "Overlapping task vectors — interference",
            font_size=24, weight=BOLD, color=RED,
        ).to_edge(UP, buff=0.5)

        lbl_a2 = MathTex(r"\tau_A", font_size=28, color=SAFETY_COLOR).shift(LEFT * 4.5 + UP * 1.3)
        lbl_b2 = MathTex(r"\tau_B", font_size=28, color=CODE_COLOR).shift(LEFT * 4.5 + DOWN * 0.1)
        lbl_sum2 = MathTex(r"\tau_A + \tau_B", font_size=24, color=WHITE).shift(LEFT * 4.5 + DOWN * 1.8)

        bars_a2 = self._bar_row(self.OVER_A, SAFETY_COLOR, y_center=1.3)
        bars_b2 = self._bar_row(self.OVER_B, CODE_COLOR, y_center=-0.1)

        the_sum = self.OVER_A + self.OVER_B
        bars_sum2 = self._bar_row(the_sum, MERGE_COLOR, y_center=-1.8)

        sep2 = DashedLine(LEFT * 4, RIGHT * 4, color=GRAY_D, stroke_width=1).shift(DOWN * 0.8)
        plus2 = MathTex("+", font_size=30).shift(LEFT * 4.5 + DOWN * 0.8)

        # Highlight the overlap region (params 4–6) with a translucent box
        bar_w = 0.3 * 1.4
        n = len(self.OVER_A)
        total_w = n * bar_w
        x_start = -total_w / 2
        # Params 4,5,6 are the overlap zone
        x_left = x_start + 4 * bar_w - bar_w * 0.3
        x_right = x_start + 6 * bar_w + bar_w * 0.3
        overlap_box = Rectangle(
            width=x_right - x_left, height=3.8,
            fill_color=RED, fill_opacity=0.1,
            stroke_color=RED, stroke_width=1.5,
        ).move_to([(x_left + x_right) / 2, -0.2, 0])

        overlap_label = Text(
            "Overlap zone:\nsign conflicts\n& doubling",
            font_size=16, color=RED, line_spacing=1.2,
        ).next_to(overlap_box, RIGHT, buff=0.3)

        self.play(Write(over_title))
        self.play(FadeIn(lbl_a2), FadeIn(bars_a2))
        self.play(FadeIn(lbl_b2), FadeIn(bars_b2))
        self.play(FadeIn(overlap_box), FadeIn(overlap_label))
        self.play(FadeIn(sep2), Write(plus2))
        self.play(FadeIn(lbl_sum2), FadeIn(bars_sum2))
        self.next_slide()

        # ── Beat 5: Vector diagram — another way to see it ───────────────────
        over_all = VGroup(
            over_title, lbl_a2, lbl_b2, lbl_sum2,
            bars_a2, bars_b2, bars_sum2, sep2, plus2,
            overlap_box, overlap_label,
        )
        self.play(FadeOut(over_all))

        vec_title = Text(
            "Another way to visualize it",
            font_size=24, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Coordinate origin at bottom-left of the visible area
        o = np.array([-4.0, -2.5, 0])

        # Shared direction: shallow diagonal going right
        shared_dir = np.array([1.0, 0.3, 0])
        shared_dir = shared_dir / np.linalg.norm(shared_dir)

        # τ_A: shared + goes steeply upward
        unique_a_dir = np.array([0.1, 1.0, 0])
        unique_a_dir = unique_a_dir / np.linalg.norm(unique_a_dir)
        a_shared = 2.0 * shared_dir
        a_unique = 2.0 * unique_a_dir
        a_total = a_shared + a_unique
        a_end = o + a_total

        # τ_B: shared + goes to the right and slightly down
        unique_b_dir = np.array([1.0, -0.5, 0])
        unique_b_dir = unique_b_dir / np.linalg.norm(unique_b_dir)
        b_shared = 1.6 * shared_dir
        b_unique = 2.0 * unique_b_dir
        b_total = b_shared + b_unique
        b_end = o + b_total

        # Origin dot
        o_dot = Dot(o, color=WHITE, radius=0.07)
        o_label = MathTex(r"\theta_{\text{pre}}", font_size=22).next_to(o_dot, DL, buff=0.1)

        # Full task vectors
        arr_a = Arrow(o, a_end, buff=0, stroke_width=4, color=SAFETY_COLOR)
        arr_b = Arrow(o, b_end, buff=0, stroke_width=4, color=CODE_COLOR)
        la = MathTex(r"\tau_A", font_size=28, color=SAFETY_COLOR).next_to(a_end, LEFT, buff=0.15)
        lb = MathTex(r"\tau_B", font_size=28, color=CODE_COLOR).next_to(b_end, DOWN, buff=0.15)

        self.play(Write(vec_title))
        self.play(FadeIn(o_dot), FadeIn(o_label))
        self.play(GrowArrow(arr_a), FadeIn(la), GrowArrow(arr_b), FadeIn(lb))
        self.next_slide()

        # Decompose: show one shared direction, then unique tails
        # Use the longer shared vector for the reference arrow
        a_sh_end = o + a_shared
        b_sh_end = o + b_shared

        # One shared-direction arrow (use τ_A's shared length)
        arr_shared = Arrow(o, a_sh_end, buff=0, stroke_width=4, color=RED)
        shared_label = Text("shared direction", font_size=18, color=RED).next_to(
            arr_shared.get_center(), DOWN, buff=0.3,
        )

        # Unique tails from each shared endpoint to the full vector tip
        arr_a_uniq = Arrow(a_sh_end, a_end, buff=0, stroke_width=3, color=SAFETY_COLOR)
        arr_b_uniq = Arrow(b_sh_end, b_end, buff=0, stroke_width=3, color=CODE_COLOR)

        unique_a_label = Text("unique A", font_size=16, color=SAFETY_COLOR).next_to(
            arr_a_uniq.get_center(), LEFT, buff=0.25,
        )
        unique_b_label = Text("unique B", font_size=16, color=CODE_COLOR).next_to(
            arr_b_uniq.get_center(), RIGHT, buff=0.25,
        )

        self.play(
            arr_a.animate.set_opacity(0.12), la.animate.set_opacity(0.2),
            arr_b.animate.set_opacity(0.12), lb.animate.set_opacity(0.2),
        )
        self.play(GrowArrow(arr_shared), FadeIn(shared_label))
        self.play(
            GrowArrow(arr_a_uniq), FadeIn(unique_a_label),
            GrowArrow(arr_b_uniq), FadeIn(unique_b_label),
        )
        self.next_slide()

        # Show sum: shared components double
        sum_shared = a_shared + b_shared
        sum_total = a_total + b_total
        sum_end = o + sum_total
        sum_sh_end = o + sum_shared

        # Clear decomposition, show the sum
        decomp = VGroup(
            arr_shared, shared_label, arr_a_uniq, arr_b_uniq,
            unique_a_label, unique_b_label,
        )
        self.play(FadeOut(decomp))

        arr_sum_sh = Arrow(o, sum_sh_end, buff=0, stroke_width=5, color=RED)
        arr_sum_rest = Arrow(sum_sh_end, sum_end, buff=0, stroke_width=3, color=MERGE_COLOR)

        sum_label = MathTex(
            r"\tau_A + \tau_B", font_size=26, color=MERGE_COLOR,
        ).next_to(sum_end, UP, buff=0.15)

        doubled_label = Text(
            "shared component doubled!",
            font_size=22, color=RED, weight=BOLD,
        ).to_edge(DOWN, buff=0.5)

        self.play(GrowArrow(arr_sum_sh), GrowArrow(arr_sum_rest), FadeIn(sum_label))
        self.play(FadeIn(doubled_label))
        self.next_slide()

        # ── Beat 6: Why it works anyway ──────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        why_title = Text(
            "Why it works well enough despite this:",
            font_size=26, weight=BOLD,
        )
        why_1 = Text(
            "1. Fine-tuning makes small perturbations\n"
            "   relative to pretrained weights\n"
            "   → approximate linearity holds.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        why_2 = Text(
            "2. Pretraining organizes weight space so\n"
            "   different tasks use different parameter subsets.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        why_punchline = Text(
            "Task vectors are approximately orthogonal\n"
            "→ cross-terms are small.",
            font_size=22, weight=BOLD, line_spacing=1.4,
        )

        why_group = VGroup(why_title, why_1, why_2, why_punchline).arrange(
            DOWN, buff=0.4, aligned_edge=LEFT,
        ).shift(UP * 0.3)

        wd_cite = Text(
            "Weight Disentanglement — Ortiz-Jimenez et al., NeurIPS 2023 (Oral)",
            font_size=18, color=GRAY_C,
        ).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(why_group, shift=UP * 0.2))
        self.play(FadeIn(wd_cite))
        self.next_slide()

        # ── Beat 7: Four failure modes + transition ──────────────────────────
        self.play(FadeOut(why_group), FadeOut(wd_cite))

        fm_title = Text(
            "Failure modes that merging methods target:",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.6)

        modes = [
            ("1. Sign conflicts", "Same parameter pushed +/− by different tasks", RED),
            ("2. Redundant updates", "Tiny noisy deltas accumulate into interference", YELLOW),
            ("3. Magnitude imbalance", "One task's vector dominates another's", ORANGE),
            ("4. Subspace misalignment", "Overlapping subspaces mix unrelated structure", MERGE_COLOR),
        ]

        mode_groups = VGroup()
        for name, desc, color in modes:
            name_text = Text(name, font_size=22, weight=BOLD, color=color)
            desc_text = Text(desc, font_size=18, color=GRAY_B)
            desc_text.next_to(name_text, DOWN, buff=0.08, aligned_edge=LEFT)
            mode_groups.add(VGroup(name_text, desc_text))

        mode_groups.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        mode_groups.next_to(fm_title, DOWN, buff=0.5, aligned_edge=LEFT)

        transition = Text(
            "Each failure mode motivates a specific family of merging methods.\nThat's what we cover next.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(fm_title))
        for mg in mode_groups:
            self.play(FadeIn(mg, shift=UP * 0.15), run_time=0.8)
        self.next_slide()

        self.play(FadeIn(transition, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.1: Stochastic Weight Averaging (§6.1) ──────────────────────────

class SWASlide(Slide):
    """SWA: the origin story of weight averaging."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Stochastic Weight Averaging", font_size=44, weight=BOLD)
        paper = Text(
            "Izmailov et al., 2018",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.3)
        paper_full = Text(
            '"Averaging Weights Leads to Wider Optima\n'
            'and Better Generalization"',
            font_size=18, color=GRAY_C, slant=ITALIC, line_spacing=1.3,
        ).next_to(paper, DOWN, buff=0.2)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.play(FadeIn(paper_full, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 2: The mechanism — cyclical LR + checkpoint averaging ───────
        self.play(FadeOut(title), FadeOut(paper), FadeOut(paper_full))

        mech_title = Text(
            "The mechanism",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        # Show the reference image (LR schedule + averaging diagram)
        mech_img = ImageMobject("assets/swa_mechanism.png")
        mech_img.scale_to_fit_width(9)
        mech_img.next_to(mech_title, DOWN, buff=0.3)

        self.play(Write(mech_title))
        self.play(FadeIn(mech_img))
        self.next_slide()

        # ── Beat 3: Key ideas as text ────────────────────────────────────────
        self.play(FadeOut(mech_img), FadeOut(mech_title))

        ideas_title = Text("Key ideas", font_size=28, weight=BOLD).to_edge(
            UP, buff=0.4,
        )

        idea_1 = Text(
            "Instead of using the final checkpoint, average\n"
            "several checkpoints from late in training.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        idea_2 = Text(
            "Use a cyclical or high-constant learning rate\n"
            "so SGD explores different points in the basin.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )

        formula = MathTex(
            r"\theta_{\text{SWA}} \leftarrow "
            r"\frac{\theta_{\text{SWA}} \cdot n + \theta}{n + 1}",
            font_size=38,
        )

        idea_3 = Text(
            "Not merging in the modern sense — it's a\n"
            "training-time technique on a single model.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )

        ideas = VGroup(idea_1, idea_2, formula, idea_3).arrange(
            DOWN, buff=0.4,
        ).next_to(ideas_title, DOWN, buff=0.5)

        self.play(Write(ideas_title))
        self.play(FadeIn(idea_1, shift=UP * 0.1))
        self.play(FadeIn(idea_2, shift=UP * 0.1))
        self.play(Write(formula))
        self.play(FadeIn(idea_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: Loss landscape — SWA vs SGD ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        land_title = Text(
            "SWA finds wider, flatter minima",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        land_img = ImageMobject("assets/swa_landscape.png")
        land_img.scale_to_fit_width(10)
        land_img.next_to(land_title, DOWN, buff=0.3)

        self.play(Write(land_title))
        self.play(FadeIn(land_img))
        self.next_slide()


# ─── Slide 6.2: Model Soups (§6.2) ──────────────────────────────────────────

class ModelSoupsSlide(Slide):
    """Model Soups: post-hoc averaging of fine-tuned models."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Model Soups", font_size=48, weight=BOLD)
        paper = Text(
            "Wortsman et al., 2022  (~800 citations)",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.3)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(paper))

        # ── Beat 2: The setup ────────────────────────────────────────────────
        setup_title = Text(
            "The idea", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        setup = Text(
            "Take a pretrained model. Fine-tune it N times\n"
            "with different hyperparameters (learning rate,\n"
            "augmentation, etc.).\n\n"
            "Then average the resulting weights.",
            font_size=22, color=GRAY_B, line_spacing=1.4,
        ).next_to(setup_title, DOWN, buff=0.5)

        formula = MathTex(
            r"\theta_{\text{soup}} = \frac{1}{N} \sum_{i=1}^{N} \theta_i",
            font_size=40,
        ).next_to(setup, DOWN, buff=0.5)

        no_train = Text(
            "No additional training. Just average and deploy.",
            font_size=20, color=YELLOW,
        ).next_to(formula, DOWN, buff=0.4)

        self.play(Write(setup_title))
        self.play(FadeIn(setup, shift=UP * 0.1))
        self.play(Write(formula))
        self.play(FadeIn(no_train, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: Types of soups ───────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        types_title = Text(
            "Three kinds of soup",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        soup_uniform_label = Text(
            "1. Uniform", font_size=22, weight=BOLD, color=SAFETY_COLOR,
        )
        soup_uniform = Text(
            "Average all N models equally.\n"
            "Simple but can be dragged down by bad runs.",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )

        soup_greedy_label = Text(
            "2. Greedy", font_size=22, weight=BOLD, color=CODE_COLOR,
        )
        soup_greedy = Text(
            "Add models one at a time. Keep each addition\n"
            "only if it improves held-out accuracy.",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )

        soup_learned_label = Text(
            "3. Learned", font_size=22, weight=BOLD, color=RLVR_COLOR,
        )
        soup_learned = Text(
            "Learn mixing coefficients per model\n"
            "(or per model per layer).",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )

        # Layout: left-aligned groups, centered on screen
        col1 = VGroup(soup_uniform_label, soup_uniform).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        col2 = VGroup(soup_greedy_label, soup_greedy).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        col3 = VGroup(soup_learned_label, soup_learned).arrange(DOWN, buff=0.1, aligned_edge=LEFT)

        cols = VGroup(col1, col2, col3).arrange(
            DOWN, buff=0.35, aligned_edge=LEFT,
        ).next_to(types_title, DOWN, buff=0.5)
        cols.shift(-cols.get_center()[0] * RIGHT)  # center the whole block horizontally

        self.play(Write(types_title))
        self.play(FadeIn(col1, shift=UP * 0.1))
        self.play(FadeIn(col2, shift=UP * 0.1))
        self.play(FadeIn(col3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: ImageNet results ─────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        res_title = Text(
            "Results: ImageNet & distribution shifts",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        res_img = ImageMobject("assets/model_soups_imagenet.png")
        res_img.scale_to_fit_width(10)
        res_img.next_to(res_title, DOWN, buff=0.3)

        self.play(Write(res_title))
        self.play(FadeIn(res_img))
        self.next_slide()

        # ── Beat 5: GLUE results ─────────────────────────────────────────────
        self.play(FadeOut(res_img), FadeOut(res_title))

        glue_title = Text(
            "Also works for NLP (GLUE benchmark)",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        glue_img = ImageMobject("assets/model_soups_glue.png")
        glue_img.scale_to_fit_width(10)
        glue_img.next_to(glue_title, DOWN, buff=0.3)

        self.play(Write(glue_title))
        self.play(FadeIn(glue_img))
        self.next_slide()

        # ── Beat 6: Why it matters ───────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        why_title = Text(
            "Why Model Soups matters",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        why_1 = Text(
            "First proof that post-hoc averaging of\n"
            "independently fine-tuned models works.",
            font_size=22, line_spacing=1.4,
        )
        why_2 = Text(
            "The loss basin around a pretrained model is\n"
            "wide enough for averaging to work across\n"
            "diverse fine-tuning runs.",
            font_size=22, color=GRAY_B, line_spacing=1.4,
        )

        why_group = VGroup(why_1, why_2).arrange(
            DOWN, buff=0.5,
        ).next_to(why_title, DOWN, buff=0.5)

        self.play(Write(why_title))
        self.play(FadeIn(why_1, shift=UP * 0.1))
        self.play(FadeIn(why_2, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.4: RegMean (§6.4) ──────────────────────────────────────────────

class RegMeanSlide(Slide):
    """RegMean: closed-form optimal merging for linear layers."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("RegMean", font_size=48, weight=BOLD)
        paper = Text(
            "Jin et al., ICLR 2023",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.25)
        full_name = Text(
            '"Dataless Knowledge Fusion by Merging\n'
            'Weights of Language Models"',
            font_size=18, color=GRAY_C, slant=ITALIC, line_spacing=1.3,
        ).next_to(paper, DOWN, buff=0.2)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.play(FadeIn(full_name, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 2: Motivation ───────────────────────────────────────────────
        self.play(FadeOut(title), FadeOut(paper), FadeOut(full_name))

        mot_title = Text(
            "The question", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        mot_1 = Text(
            "Model Soups: average all parameters equally.",
            font_size=21, color=GRAY_B,
        )
        mot_2 = Text(
            "Fisher Merging: weight by importance per parameter.",
            font_size=21, color=GRAY_B,
        )
        mot_3 = Text(
            "What if we could compute the mathematically\n"
            "optimal merge for each layer?",
            font_size=24, weight=BOLD, color=YELLOW, line_spacing=1.4,
        )

        mot_group = VGroup(mot_1, mot_2, mot_3).arrange(
            DOWN, buff=0.5,
        ).next_to(mot_title, DOWN, buff=0.5)

        self.play(Write(mot_title))
        self.play(FadeIn(mot_1, shift=UP * 0.1))
        self.play(FadeIn(mot_2, shift=UP * 0.1))
        self.play(FadeIn(mot_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: The insight — linear layers as regression ────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        insight_title = Text(
            "The insight", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Show what a linear layer does
        layer_eq = MathTex(
            r"y = W \, x",
            font_size=40,
        )
        layer_explain = Text(
            "A linear layer: input x, weight matrix W, output y.\n"
            "(Attention projections, FFN layers — all linear.)",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        # The question posed as regression
        question = Text(
            "Given N models with weights W₁, W₂, ..., Wₙ,\n"
            "find a single W that best reproduces what\n"
            "each model would have predicted on its own inputs.",
            font_size=21, line_spacing=1.4,
        )

        regression = Text(
            "This is just a least-squares regression problem.",
            font_size=22, weight=BOLD, color=GREEN,
        )

        insight_group = VGroup(layer_eq, layer_explain, question, regression).arrange(
            DOWN, buff=0.4,
        ).next_to(insight_title, DOWN, buff=0.4)

        self.play(Write(insight_title))
        self.play(Write(layer_eq))
        self.play(FadeIn(layer_explain, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeIn(question, shift=UP * 0.1))
        self.play(FadeIn(regression, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: The objective ────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        obj_title = Text(
            "The objective", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Objective function
        objective = MathTex(
            r"W^* = \arg\min_W \sum_{i=1}^{N}",
            r"\| W \, X_i - W_i \, X_i \|^2_F",
            font_size=34,
        )

        obj_explain = VGroup(
            Text("For each model i:", font_size=20, weight=BOLD),
            MathTex(r"W_i", font_size=28, color=SAFETY_COLOR).shift(LEFT * 0.5),
            Text("= that model's learned weights", font_size=19, color=GRAY_B),
        ).arrange(RIGHT, buff=0.15)

        obj_explain2 = VGroup(
            MathTex(r"X_i", font_size=28, color=CODE_COLOR).shift(LEFT * 0.5),
            Text("= the inputs that model i saw during training", font_size=19, color=GRAY_B),
        ).arrange(RIGHT, buff=0.15)

        obj_explain3 = VGroup(
            MathTex(r"\| \cdot \|^2_F", font_size=28).shift(LEFT * 0.5),
            Text("= sum of squared differences (Frobenius norm)", font_size=19, color=GRAY_B),
        ).arrange(RIGHT, buff=0.15)

        obj_plain = Text(
            '"Find the W whose predictions are as close\n'
            'as possible to each individual model\'s predictions,\n'
            'on that model\'s own data."',
            font_size=20, color=YELLOW, slant=ITALIC, line_spacing=1.4,
        )

        obj_group = VGroup(
            objective, obj_explain, obj_explain2, obj_explain3, obj_plain,
        ).arrange(DOWN, buff=0.35).next_to(obj_title, DOWN, buff=0.4)

        self.play(Write(obj_title))
        self.play(Write(objective))
        self.next_slide()
        self.play(FadeIn(obj_explain, shift=UP * 0.1))
        self.play(FadeIn(obj_explain2, shift=UP * 0.1))
        self.play(FadeIn(obj_explain3, shift=UP * 0.1))
        self.play(FadeIn(obj_plain, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: The closed-form solution ─────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        sol_title = Text(
            "The closed-form solution", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        solution = MathTex(
            r"W^* = \left( \sum_{i=1}^{N} X_i^\top X_i \right)^{-1}"
            r"\left( \sum_{i=1}^{N} X_i^\top X_i \, W_i \right)",
            font_size=32,
        )

        # Explain each term
        term1_label = MathTex(
            r"X_i^\top X_i", font_size=30, color=CODE_COLOR,
        )
        term1_text = Text(
            '= input covariance of model i.\n'
            '  Captures "which input directions does this\n'
            '  task care about?" If a task activates certain\n'
            '  neurons heavily, those directions get more weight.',
            font_size=18, color=GRAY_B, line_spacing=1.3,
        )
        term1 = VGroup(term1_label, term1_text).arrange(RIGHT, buff=0.2, aligned_edge=UP)

        term2_label = MathTex(
            r"X_i^\top X_i \, W_i", font_size=30, color=SAFETY_COLOR,
        )
        term2_text = Text(
            '= model i\'s weights, weighted by how much\n'
            '  its data "uses" each input direction.\n'
            '  Important parameters get amplified.',
            font_size=18, color=GRAY_B, line_spacing=1.3,
        )
        term2 = VGroup(term2_label, term2_text).arrange(RIGHT, buff=0.2, aligned_edge=UP)

        sol_group = VGroup(solution, term1, term2).arrange(
            DOWN, buff=0.4,
        ).next_to(sol_title, DOWN, buff=0.4)

        self.play(Write(sol_title))
        self.play(Write(solution))
        self.next_slide()
        self.play(FadeIn(term1, shift=UP * 0.1))
        self.play(FadeIn(term2, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 6: Intuition + properties ───────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        prop_title = Text(
            "In plain English", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        intuition = Text(
            "Simple averaging treats every parameter equally.\n"
            "RegMean asks: how much did each model actually\n"
            "use this parameter? And weights accordingly.",
            font_size=22, line_spacing=1.4,
        )

        props = VGroup(
            Text("Properties:", font_size=22, weight=BOLD),
            Text("  Closed-form — no iterative optimization", font_size=20, color=GRAY_B),
            Text("  No hyperparameters to tune", font_size=20, color=GRAY_B),
            Text("  Applied independently to each linear layer", font_size=20, color=GRAY_B),
            Text("  Only needs input statistics, not training data", font_size=20, color=GRAY_B),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        limitation = Text(
            "Limitation: merges each layer independently.\n"
            "Ignores how layers interact with each other.",
            font_size=20, color=GRAY_C, line_spacing=1.3,
        )

        prop_group = VGroup(intuition, props, limitation).arrange(
            DOWN, buff=0.45,
        ).next_to(prop_title, DOWN, buff=0.4)

        self.play(Write(prop_title))
        self.play(FadeIn(intuition, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeIn(props, shift=UP * 0.1))
        self.play(FadeIn(limitation, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.3: Fisher Merging (§6.3) ───────────────────────────────────────

class FisherMergingSlide(Slide):
    """Fisher Merging: importance-weighted parameter averaging."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Fisher Merging", font_size=48, weight=BOLD)
        paper = Text(
            "Matena & Raffel, NeurIPS 2022  (~500 citations)",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.3)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(paper))

        # ── Beat 2: The problem with uniform averaging ───────────────────────
        prob_title = Text(
            "The problem with uniform averaging",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        prob_1 = Text(
            "Simple averaging treats all parameters equally.",
            font_size=22,
        )
        prob_2 = Text(
            "But some parameters matter more than others.",
            font_size=22, color=YELLOW, weight=BOLD,
        )
        prob_3 = Text(
            "A parameter critical for Task A but irrelevant\n"
            "for Task B should be weighted toward A's value.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        prob_group = VGroup(prob_1, prob_2, prob_3).arrange(
            DOWN, buff=0.4,
        ).next_to(prob_title, DOWN, buff=0.5)

        self.play(Write(prob_title))
        self.play(FadeIn(prob_1, shift=UP * 0.1))
        self.play(FadeIn(prob_2, shift=UP * 0.1))
        self.play(FadeIn(prob_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: What is Fisher information? (Intuition) ──────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        fi_title = Text(
            "What is Fisher information?",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        fi_1 = Text(
            "How much does the model's output change\n"
            "if you wiggle a parameter?",
            font_size=22, line_spacing=1.4,
        )
        fi_high = Text(
            "High Fisher  →  model is sensitive\n"
            "                           →  parameter matters a lot",
            font_size=20, color=GREEN, line_spacing=1.3,
        )
        fi_low = Text(
            "Low Fisher   →  can change freely\n"
            "                          →  doesn't matter much",
            font_size=20, color=RED, line_spacing=1.3,
        )
        fi_formal = MathTex(
            r"F_j = \mathbb{E}\!\left[\left(\frac{\partial \log p(x \mid \theta)}{\partial \theta_j}\right)^{\!2}\right]",
            font_size=34,
        )
        fi_gloss = Text(
            "Expected squared gradient of the log-likelihood\n"
            "with respect to each parameter.",
            font_size=18, color=GRAY_C, line_spacing=1.3,
        )

        fi_group = VGroup(fi_1, fi_high, fi_low, fi_formal, fi_gloss).arrange(
            DOWN, buff=0.35,
        ).next_to(fi_title, DOWN, buff=0.4)

        self.play(Write(fi_title))
        self.play(FadeIn(fi_1, shift=UP * 0.1))
        self.play(FadeIn(fi_high, shift=UP * 0.1))
        self.play(FadeIn(fi_low, shift=UP * 0.1))
        self.play(Write(fi_formal))
        self.play(FadeIn(fi_gloss, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: Visual — parameter importance bars ───────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        bar_title = Text(
            "Most parameters don't matter. A few matter a lot.",
            font_size=24, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Bar chart of Fisher values for ~10 parameters
        fisher_vals = [0.05, 0.9, 0.08, 0.03, 0.7, 0.04, 0.02, 0.85, 0.06, 0.03]
        bars = VGroup()
        labels = VGroup()
        bar_w = 0.4
        max_h = 3.0
        for i, v in enumerate(fisher_vals):
            h = v * max_h
            bar = Rectangle(
                width=bar_w, height=max(h, 0.05),
                fill_color=GREEN if v > 0.5 else GRAY_D,
                fill_opacity=0.8 if v > 0.5 else 0.4,
                stroke_width=0.5, stroke_color=WHITE,
            )
            lbl = MathTex(f"\\theta_{{{i+1}}}", font_size=16)
            bars.add(bar)
            labels.add(lbl)

        bars.arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        bars.next_to(bar_title, DOWN, buff=0.8)

        for bar, lbl in zip(bars, labels):
            lbl.next_to(bar, DOWN, buff=0.1)

        fisher_label = Text(
            "Fisher value (importance)", font_size=18, color=GRAY_C,
        ).next_to(bars, LEFT, buff=0.5)

        high_label = Text("important", font_size=16, color=GREEN)
        low_label = Text("unimportant", font_size=16, color=GRAY_D)
        high_label.next_to(bars, UP, buff=0.2).shift(RIGHT * 0.5)
        low_label.next_to(bars, UP, buff=0.2).shift(LEFT * 2)

        self.play(Write(bar_title))
        self.play(
            *[FadeIn(b, shift=UP * 0.1) for b in bars],
            *[FadeIn(l) for l in labels],
        )
        self.play(FadeIn(fisher_label), FadeIn(high_label), FadeIn(low_label))
        self.next_slide()

        # ── Beat 5: Derivation reference (dense) ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        deriv_title = Text(
            "Derivation (reference)",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.3)

        # Step-by-step derivation as a single block of LaTeX + text
        step1_label = Text("1. Goal: find θ that maximizes joint posterior", font_size=17, color=GRAY_B)
        step1_math = MathTex(
            r"\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log p(\theta \mid \mathcal{D}_i)",
            font_size=28,
        )

        step2_label = Text("2. Laplace approx: each posterior is Gaussian around θ_i", font_size=17, color=GRAY_B)
        step2_math = MathTex(
            r"\log p(\theta \mid \mathcal{D}_i) \approx"
            r" -\tfrac{1}{2}(\theta - \theta_i)^\top F_i (\theta - \theta_i) + \text{const}",
            font_size=26,
        )

        step3_label = Text("3. Sum the log-posteriors, take derivative, set to zero", font_size=17, color=GRAY_B)
        step3_math = MathTex(
            r"\sum_{i=1}^{N} F_i (\theta - \theta_i) = 0",
            font_size=28,
        )

        step4_label = Text("4. Solve for θ (diagonal approximation: F_i → diag)", font_size=17, color=GRAY_B)
        step4_math = MathTex(
            r"\theta^*_j = \frac{\sum_{i=1}^{N} F_{i,j} \, \theta_{i,j}}"
            r"{\sum_{i=1}^{N} F_{i,j}}",
            font_size=30,
        )

        deriv_steps = VGroup(
            step1_label, step1_math,
            step2_label, step2_math,
            step3_label, step3_math,
            step4_label, step4_math,
        ).arrange(DOWN, buff=0.15).next_to(deriv_title, DOWN, buff=0.3)

        # Show everything at once (reference slide)
        self.play(FadeIn(deriv_title), FadeIn(deriv_steps), run_time=0.5)
        self.next_slide()

        # ── Beat 7: The formula (clean) ──────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        form_title = Text(
            "The Fisher Merging formula",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        formula = MathTex(
            r"\theta^*_j = \frac{\sum_{i} F_{i,j} \, \theta_{i,j}}"
            r"{\sum_{i} F_{i,j}}",
            font_size=44,
        )

        gloss = Text(
            "Each parameter is a weighted average,\n"
            "weighted by how important it is to each model.",
            font_size=22, color=GRAY_B, line_spacing=1.4,
        ).next_to(formula, DOWN, buff=0.5)

        form_group = VGroup(formula, gloss).next_to(form_title, DOWN, buff=0.8)

        self.play(Write(form_title))
        self.play(Write(formula), run_time=1.5)
        self.play(FadeIn(gloss, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 8: Key insight ──────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        insight = Text(
            "Not all parameters matter equally\n"
            "during merging.",
            font_size=32, weight=BOLD, line_spacing=1.4,
        )
        insight_sub = Text(
            "This idea reappears in every subsequent method.",
            font_size=22, color=GRAY_B,
        ).next_to(insight, DOWN, buff=0.5)

        self.play(FadeIn(insight, scale=1.1))
        self.play(FadeIn(insight_sub, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 9: Limitations ──────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        lim_title = Text(
            "Limitations",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        lim_1 = Text(
            "Computing the full Fisher is expensive: O(params²).",
            font_size=21, color=GRAY_B,
        )
        lim_2 = Text(
            "In practice: diagonal approximation\n"
            "(one number per parameter).",
            font_size=21, color=GRAY_B, line_spacing=1.3,
        )
        lim_3 = Text(
            "Outperformed by simpler methods (TIES, DARE)\n"
            "on most benchmarks.",
            font_size=21, color=GRAY_B, line_spacing=1.3,
        )
        lim_4 = Text(
            "But the idea was foundational.",
            font_size=22, weight=BOLD,
        )

        lim_group = VGroup(lim_1, lim_2, lim_3, lim_4).arrange(
            DOWN, buff=0.35,
        ).next_to(lim_title, DOWN, buff=0.5)

        self.play(Write(lim_title))
        self.play(FadeIn(lim_1, shift=UP * 0.1))
        self.play(FadeIn(lim_2, shift=UP * 0.1))
        self.play(FadeIn(lim_3, shift=UP * 0.1))
        self.play(FadeIn(lim_4, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.6: TIES-Merging (§6.6) ─────────────────────────────────────────

class TIESSlide(Slide):
    """TIES-Merging: Trim, Elect sign, Merge agreeing parameters."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("TIES-Merging", font_size=48, weight=BOLD)
        paper = Text(
            "Yadav et al., NeurIPS 2023  (~400 citations)",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.25)
        full_name = Text(
            '"Resolving Interference When Merging Models"',
            font_size=18, color=GRAY_C, slant=ITALIC,
        ).next_to(paper, DOWN, buff=0.2)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.play(FadeIn(full_name, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 2: The two problems ─────────────────────────────────────────
        self.play(FadeOut(title), FadeOut(paper), FadeOut(full_name))

        prob_title = Text(
            "Two sources of interference", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        prob_1_name = Text(
            "1. Redundant parameters", font_size=22, weight=BOLD, color=YELLOW,
        )
        prob_1_desc = Text(
            "Most parameters barely change during fine-tuning.\n"
            "These near-zero deltas are noise — but when you\n"
            "average them across many models, the noise adds up\n"
            "and dilutes the signal from parameters that actually matter.",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )
        prob_1 = VGroup(prob_1_name, prob_1_desc).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        prob_2_name = Text(
            "2. Sign disagreements", font_size=22, weight=BOLD, color=RED,
        )
        prob_2_desc = Text(
            "For the same parameter, one model pushes it positive\n"
            "while another pushes it negative. Averaging them\n"
            "cancels both signals — you lose what both models learned.",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )
        prob_2 = VGroup(prob_2_name, prob_2_desc).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        probs = VGroup(prob_1, prob_2).arrange(
            DOWN, buff=0.4, aligned_edge=LEFT,
        ).next_to(prob_title, DOWN, buff=0.5).shift(LEFT * 1.5)

        self.play(Write(prob_title))
        self.play(FadeIn(prob_1, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeIn(prob_2, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: The TIES acronym ─────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        ties_title = Text(
            "TIES: a three-step pipeline", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Acronym breakdown
        t_label = Text("Tr I m", font_size=36, weight=BOLD, color=SAFETY_COLOR)
        i_label = Text("  E lect", font_size=36, weight=BOLD, color=CODE_COLOR)
        s_label = Text("  S ign &  Merge", font_size=36, weight=BOLD, color=RLVR_COLOR)

        # Simpler: just show the three steps as a clean list
        step1 = VGroup(
            Text("1. Trim", font_size=28, weight=BOLD, color=SAFETY_COLOR),
            Text("Drop low-magnitude deltas", font_size=20, color=GRAY_B),
        ).arrange(RIGHT, buff=0.3)
        step2 = VGroup(
            Text("2. Elect Sign", font_size=28, weight=BOLD, color=CODE_COLOR),
            Text("Majority vote on direction", font_size=20, color=GRAY_B),
        ).arrange(RIGHT, buff=0.3)
        step3 = VGroup(
            Text("3. Disjoint Merge", font_size=28, weight=BOLD, color=RLVR_COLOR),
            Text("Average only the agreeing params", font_size=20, color=GRAY_B),
        ).arrange(RIGHT, buff=0.3)

        steps = VGroup(step1, step2, step3).arrange(
            DOWN, buff=0.5, aligned_edge=LEFT,
        ).next_to(ties_title, DOWN, buff=0.6)

        self.play(Write(ties_title))
        self.play(FadeIn(step1, shift=UP * 0.1))
        self.play(FadeIn(step2, shift=UP * 0.1))
        self.play(FadeIn(step3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3b: Pipeline diagram from the paper ─────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        diag_title = Text(
            "The full pipeline (from the paper)",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        diag_img = ImageMobject("assets/ties_pipeline.png")
        diag_img.scale_to_fit_width(11)
        diag_img.next_to(diag_title, DOWN, buff=0.3)

        self.play(Write(diag_title))
        self.play(FadeIn(diag_img))
        self.next_slide()

        # ── Beat 4: Step 1 — Trim ───────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        trim_title = Text(
            "Step 1: Trim", font_size=30, weight=BOLD, color=SAFETY_COLOR,
        ).to_edge(UP, buff=0.5)

        trim_formula = MathTex(
            r"\tilde{\tau}_{i,p} = \tau_{i,p} \cdot \mathbb{1}"
            r"\big( |\tau_{i,p}| \geq \text{threshold} \big)",
            font_size=30,
        )

        trim_explain = Text(
            "For each parameter p in each task vector τᵢ:\n"
            "keep it if its magnitude is above a threshold,\n"
            "otherwise zero it out.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        trim_intuition = Text(
            "In practice, keep only the top k% of parameters\n"
            "by magnitude. The rest are noise.",
            font_size=20, line_spacing=1.4,
        )

        density = Text(
            'The "density" hyperparameter controls k%.\n'
            "density = 0.2 → keep top 20% (aggressive)\n"
            "density = 0.5 → keep top 50% (default)\n"
            "density = 1.0 → keep everything (no trimming)",
            font_size=18, color=GRAY_C, line_spacing=1.3,
        )

        trim_group = VGroup(trim_formula, trim_explain, trim_intuition, density).arrange(
            DOWN, buff=0.4,
        ).next_to(trim_title, DOWN, buff=0.4)

        self.play(Write(trim_title))
        self.play(Write(trim_formula))
        self.play(FadeIn(trim_explain, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeIn(trim_intuition, shift=UP * 0.1))
        self.play(FadeIn(density, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: Step 2 — Elect Sign ─────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        elect_title = Text(
            "Step 2: Elect Sign", font_size=30, weight=BOLD, color=CODE_COLOR,
        ).to_edge(UP, buff=0.5)

        elect_formula = MathTex(
            r"s_p = \text{sgn}\left( \sum_{i=1}^{N} \tilde{\tau}_{i,p} \right)",
            font_size=34,
        )

        elect_explain = Text(
            "For each parameter p, sum the trimmed deltas\n"
            "across all models. The sign of the sum tells you\n"
            "which direction the majority of models agree on.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        elect_example = Text(
            "Example: 3 models want parameter p to go positive,\n"
            "1 model wants it negative → elected sign is +.",
            font_size=20, line_spacing=1.4,
        )

        elect_key = Text(
            "This is a parameter-wise majority vote\n"
            "on the direction of change.",
            font_size=22, weight=BOLD, color=CODE_COLOR, line_spacing=1.4,
        )

        elect_group = VGroup(elect_formula, elect_explain, elect_example, elect_key).arrange(
            DOWN, buff=0.4,
        ).next_to(elect_title, DOWN, buff=0.4)

        self.play(Write(elect_title))
        self.play(Write(elect_formula))
        self.play(FadeIn(elect_explain, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeIn(elect_example, shift=UP * 0.1))
        self.play(FadeIn(elect_key, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 6: Step 3 — Disjoint Merge ─────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        merge_title = Text(
            "Step 3: Disjoint Merge", font_size=30, weight=BOLD, color=RLVR_COLOR,
        ).to_edge(UP, buff=0.5)

        merge_explain_1 = Text(
            "For each parameter p:",
            font_size=22, weight=BOLD,
        )
        merge_explain_2 = Text(
            "Look at only the models whose trimmed delta\n"
            "agrees with the elected sign sₚ.\n"
            "Discard the rest.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        merge_explain_3 = Text(
            "Average only the agreeing deltas.",
            font_size=22, weight=BOLD, color=RLVR_COLOR,
        )

        merge_example = Text(
            "If 3 models agree on + and 1 disagrees (−):\n"
            "→ drop the dissenter\n"
            "→ average the 3 positive deltas\n"
            "→ the negative signal doesn't cancel them out",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )

        final_formula = MathTex(
            r"\theta_{\text{merged}} = \theta_{\text{pre}} + \lambda \cdot \tilde{\tau}_{\text{TIES}}",
            font_size=34,
        )
        final_note = Text(
            "λ is a scaling factor (like Task Arithmetic).\n"
            "The merged task vector is cleaner because\n"
            "noise was trimmed and conflicts were resolved.",
            font_size=19, color=GRAY_C, line_spacing=1.3,
        )

        merge_group = VGroup(
            merge_explain_1, merge_explain_2, merge_explain_3,
            merge_example, final_formula, final_note,
        ).arrange(DOWN, buff=0.3).next_to(merge_title, DOWN, buff=0.4)

        self.play(Write(merge_title))
        self.play(FadeIn(merge_explain_1, shift=UP * 0.1))
        self.play(FadeIn(merge_explain_2, shift=UP * 0.1))
        self.play(FadeIn(merge_explain_3, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeIn(merge_example, shift=UP * 0.1))
        self.play(Write(final_formula))
        self.play(FadeIn(final_note, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 7: Why TIES matters ─────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        why_title = Text(
            "Why TIES matters", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        why_1 = Text(
            "Defined the vocabulary the field still uses:\n"
            "trim, elect, disjoint merge.",
            font_size=21, line_spacing=1.4,
        )
        why_2 = Text(
            "Extremely lightweight — no training data needed,\n"
            "no optimization, just arithmetic on task vectors.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        why_3 = Text(
            "Directly addresses the failure modes from §4:\n"
            "trimming handles redundant updates,\n"
            "sign election handles sign conflicts.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )

        why_group = VGroup(why_1, why_2, why_3).arrange(
            DOWN, buff=0.45,
        ).next_to(why_title, DOWN, buff=0.5)

        self.play(Write(why_title))
        self.play(FadeIn(why_1, shift=UP * 0.1))
        self.play(FadeIn(why_2, shift=UP * 0.1))
        self.play(FadeIn(why_3, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.7: Subspace Boosting (§6.7) ────────────────────────────────────

class SubspaceBoostingSlide(Slide):
    """Subspace Boosting: fixing rank collapse in task arithmetic."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Subspace Boosting", font_size=48, weight=BOLD)
        paper = Text(
            "Skorobogat et al., 2025",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.3)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(paper))

        # ── Beat 2: The problem — what happens as you merge more models ──────
        prob_title = Text(
            "The problem: diminishing returns",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        prob_1 = Text(
            "Task Arithmetic merges N expert models by\n"
            "summing their task vectors:",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )

        prob_formula = MathTex(
            r"\theta_{\text{merged}} = \theta_{\text{pre}} + \sum_{i=1}^{N} \lambda_i \, \tau_i",
            font_size=36,
        )

        prob_2 = Text(
            "This works for 2–3 models.\n"
            "But as N grows, performance degrades.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )

        prob_3 = Text(
            "Why?",
            font_size=26, color=YELLOW, weight=BOLD,
        )

        prob_group = VGroup(prob_1, prob_formula, prob_2, prob_3).arrange(
            DOWN, buff=0.35,
        ).next_to(prob_title, DOWN, buff=0.4)

        self.play(Write(prob_title))
        self.play(FadeIn(prob_1, shift=UP * 0.1))
        self.play(Write(prob_formula))
        self.play(FadeIn(prob_2, shift=UP * 0.1))
        self.play(FadeIn(prob_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: Rank collapse — the theorem ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        rc_title = Text(
            "Rank collapse (a theorem, not just empirical)",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        rc_1 = Text(
            "Each task vector can be decomposed into:",
            font_size=21,
        )
        rc_formula = MathTex(
            r"\tau_i = \underbrace{\tau_{\text{common}}}_{\text{shared across tasks}}"
            r" + \underbrace{\tau_i^{\text{unique}}}_{\text{task-specific}}",
            font_size=34,
        )

        rc_2 = Text(
            "When you sum N task vectors:",
            font_size=21,
        )

        rc_scaling = VGroup(
            Text("Common information grows as  ", font_size=21),
            MathTex(r"O(N)", font_size=30, color=RED),
        ).arrange(RIGHT, buff=0.1)

        rc_scaling2 = VGroup(
            Text("Task-specific information grows as  ", font_size=21),
            MathTex(r"O(\sqrt{N})", font_size=30, color=GREEN),
        ).arrange(RIGHT, buff=0.1)

        rc_3 = Text(
            "The ratio of unique to common signal → 0 as N → ∞.\n"
            "The unique knowledge gets drowned out.",
            font_size=20, color=YELLOW, line_spacing=1.4,
        )

        rc_group = VGroup(rc_1, rc_formula, rc_2, rc_scaling, rc_scaling2, rc_3).arrange(
            DOWN, buff=0.3,
        ).next_to(rc_title, DOWN, buff=0.4)

        self.play(Write(rc_title))
        self.play(FadeIn(rc_1, shift=UP * 0.1))
        self.play(Write(rc_formula))
        self.play(FadeIn(rc_2, shift=UP * 0.1))
        self.play(FadeIn(rc_scaling, shift=UP * 0.1))
        self.play(FadeIn(rc_scaling2, shift=UP * 0.1))
        self.play(FadeIn(rc_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: Visual — singular value distribution ─────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        sv_title = Text(
            "Visualizing rank collapse via singular values",
            font_size=24, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Axes for singular value plot
        ax = Axes(
            x_range=[0, 10, 1], y_range=[0, 1.2, 0.5],
            x_length=8, y_length=3.5,
            axis_config={"include_numbers": False, "stroke_opacity": 0.5},
            tips=False,
        ).shift(DOWN * 0.5)

        x_label = Text("Singular value index", font_size=16, color=GRAY_C).next_to(
            ax, DOWN, buff=0.2,
        )
        y_label = Text("Magnitude", font_size=16, color=GRAY_C).next_to(
            ax, LEFT, buff=0.2,
        ).rotate(90 * DEGREES)

        # N=2: healthy distribution — gradual decay
        curve_2 = ax.plot(
            lambda x: 1.0 * np.exp(-0.2 * x),
            x_range=[0.5, 9.5], color=GREEN, stroke_width=3,
        )
        label_2 = Text("N = 2 models", font_size=16, color=GREEN)

        # N=10: tail collapses — sharp drop
        curve_10 = ax.plot(
            lambda x: 1.0 * np.exp(-0.6 * x),
            x_range=[0.5, 9.5], color=YELLOW, stroke_width=3,
        )
        label_10 = Text("N = 10 models", font_size=16, color=YELLOW)

        # N=20: severe collapse — only first 1-2 survive
        curve_20 = ax.plot(
            lambda x: 1.0 * np.exp(-1.5 * x),
            x_range=[0.5, 9.5], color=RED, stroke_width=3,
        )
        label_20 = Text("N = 20 models", font_size=16, color=RED)

        # Legend
        legend = VGroup(label_2, label_10, label_20).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        legend.next_to(ax, RIGHT, buff=0.4).shift(UP * 0.5)

        collapse_label = Text(
            "← task-specific info lost",
            font_size=16, color=RED,
        ).next_to(ax.c2p(6, 0.05), UP, buff=0.1)

        self.play(Write(sv_title))
        self.play(Create(ax), FadeIn(x_label), FadeIn(y_label))
        self.play(Create(curve_2), FadeIn(label_2))
        self.play(Create(curve_10), FadeIn(label_10))
        self.play(Create(curve_20), FadeIn(label_20))
        self.play(FadeIn(collapse_label))
        self.next_slide()

        # ── Beat 5: The fix — SVD + boost ────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        fix_title = Text(
            "The fix: boost the suppressed singular values",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        step_1 = Text(
            "1. Compute the merged task vector:",
            font_size=20,
        )
        step_1_math = MathTex(
            r"\tau_{\text{merged}} = \sum_{i=1}^{N} \lambda_i \tau_i",
            font_size=32,
        )

        step_2 = Text(
            "2. Decompose via SVD:",
            font_size=20,
        )
        step_2_math = MathTex(
            r"\tau_{\text{merged}} = U \Sigma V^\top",
            font_size=32,
        )

        step_3 = Text(
            "3. Boost the small singular values in Σ:",
            font_size=20,
        )
        step_3_math = MathTex(
            r"\tilde{\sigma}_k = \sigma_k^{1 - \alpha}",
            font_size=32,
        )
        step_3_note = Text(
            "α ∈ [0, 1] — the single hyperparameter.\n"
            "α = 0: no change.  α → 1: flatten all singular values.",
            font_size=17, color=GRAY_C, line_spacing=1.3,
        )

        step_4 = Text(
            "4. Reconstruct:",
            font_size=20,
        )
        step_4_math = MathTex(
            r"\tilde{\tau}_{\text{merged}} = U \tilde{\Sigma} V^\top",
            font_size=32,
        )

        fix_group = VGroup(
            step_1, step_1_math,
            step_2, step_2_math,
            step_3, step_3_math, step_3_note,
            step_4, step_4_math,
        ).arrange(DOWN, buff=0.15).next_to(fix_title, DOWN, buff=0.3)

        self.play(Write(fix_title))
        self.play(FadeIn(step_1, shift=UP * 0.1), Write(step_1_math))
        self.play(FadeIn(step_2, shift=UP * 0.1), Write(step_2_math))
        self.next_slide()
        self.play(FadeIn(step_3, shift=UP * 0.1), Write(step_3_math))
        self.play(FadeIn(step_3_note, shift=UP * 0.1))
        self.play(FadeIn(step_4, shift=UP * 0.1), Write(step_4_math))
        self.next_slide()

        # ── Beat 6: Intuition — what the boost does ──────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        int_title = Text(
            "What the boost does",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        int_1 = Text(
            "The dominant singular values capture\n"
            "common information shared across all tasks.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        int_2 = Text(
            "The small singular values capture\n"
            "task-specific information unique to each expert.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        int_3 = Text(
            "Rank collapse suppresses the small ones.\n"
            "Subspace Boosting lifts them back up.",
            font_size=22, weight=BOLD, line_spacing=1.4,
        )
        int_4 = Text(
            "It's re-balancing the spectrum so that\n"
            "unique knowledge isn't drowned out by\n"
            "what all models already agree on.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        int_group = VGroup(int_1, int_2, int_3, int_4).arrange(
            DOWN, buff=0.35,
        ).next_to(int_title, DOWN, buff=0.5)

        self.play(Write(int_title))
        self.play(FadeIn(int_1, shift=UP * 0.1))
        self.play(FadeIn(int_2, shift=UP * 0.1))
        self.play(FadeIn(int_3, shift=UP * 0.1))
        self.play(FadeIn(int_4, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 7: Results + HO-GSVD mention ───────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        res_title = Text(
            "Results & additional contributions",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        res_1 = Text(
            ">10% improvement when merging up to\n"
            "20 experts on vision and language benchmarks.",
            font_size=22, color=GREEN, line_spacing=1.4,
        )
        res_2 = Text(
            "One hyperparameter (α). Drop-in replacement\n"
            "for the summation step in Task Arithmetic.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        res_3 = Text(
            "Also introduces HO-GSVD\n"
            "(Higher-Order Generalized SVD):",
            font_size=20, weight=BOLD, line_spacing=1.3,
        )
        res_4 = Text(
            "Transforms independent task vector subspaces\n"
            "into a shared subspace. Lets you quantify task\n"
            "similarity and decide which models to merge.",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )

        res_group = VGroup(res_1, res_2, res_3, res_4).arrange(
            DOWN, buff=0.35,
        ).next_to(res_title, DOWN, buff=0.5)

        self.play(Write(res_title))
        self.play(FadeIn(res_1, shift=UP * 0.1))
        self.play(FadeIn(res_2, shift=UP * 0.1))
        self.play(FadeIn(res_3, shift=UP * 0.1))
        self.play(FadeIn(res_4, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.8: AdaMerging (§6.8) ───────────────────────────────────────────

class AdaMergingSlide(Slide):
    """AdaMerging: learn the merging coefficients automatically."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("AdaMerging", font_size=48, weight=BOLD)
        paper = Text(
            "Yang et al., ICLR 2024",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.3)
        subtitle = Text(
            "Adaptive Model Merging for Multi-Task Learning",
            font_size=18, color=GRAY_C, slant=ITALIC,
        ).next_to(paper, DOWN, buff=0.2)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.play(FadeIn(subtitle, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(paper), FadeOut(subtitle))

        # ── Beat 2: The question ─────────────────────────────────────────────
        q_title = Text(
            "The question",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        q_1 = Text(
            "Every method so far requires hand-tuning\n"
            "the merging coefficients (λ₁, λ₂, ..., λₙ).",
            font_size=22, line_spacing=1.4,
        )
        q_2 = Text(
            "What if we just learned them?",
            font_size=26, color=YELLOW, weight=BOLD,
        )
        q_3 = Text(
            "And what if we could do it without\n"
            "any labeled data?",
            font_size=22, color=GRAY_B, line_spacing=1.4,
        )

        q_group = VGroup(q_1, q_2, q_3).arrange(
            DOWN, buff=0.4,
        ).next_to(q_title, DOWN, buff=0.5)

        self.play(Write(q_title))
        self.play(FadeIn(q_1, shift=UP * 0.1))
        self.play(FadeIn(q_2, shift=UP * 0.1))
        self.play(FadeIn(q_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: Two variants ─────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        var_title = Text(
            "Two variants",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Task-wise
        tw_label = Text(
            "Task-wise AdaMerging",
            font_size=24, weight=BOLD, color=SAFETY_COLOR,
        )
        tw_desc = Text(
            "One coefficient per task.\n"
            "Learn λ₁, λ₂, ..., λₙ.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        tw_formula = MathTex(
            r"\theta = \theta_{\text{pre}} + \sum_{i=1}^{N} \lambda_i \, \tau_i",
            font_size=30,
        )
        tw = VGroup(tw_label, tw_desc, tw_formula).arrange(DOWN, buff=0.15)

        # Layer-wise
        lw_label = Text(
            "Layer-wise AdaMerging",
            font_size=24, weight=BOLD, color=CODE_COLOR,
        )
        lw_desc = Text(
            "One coefficient per task per layer.\n"
            "Learn λ₁ˡ, λ₂ˡ, ..., λₙˡ for each layer l.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        lw_formula = MathTex(
            r"\theta^{(l)} = \theta_{\text{pre}}^{(l)} + \sum_{i=1}^{N} \lambda_i^{(l)} \, \tau_i^{(l)}",
            font_size=30,
        )
        lw = VGroup(lw_label, lw_desc, lw_formula).arrange(DOWN, buff=0.15)

        lw_note = Text(
            "Much more expressive. Consistently better.",
            font_size=20, color=GREEN, weight=BOLD,
        )

        variants = VGroup(tw, lw, lw_note).arrange(
            DOWN, buff=0.4,
        ).next_to(var_title, DOWN, buff=0.4)

        self.play(Write(var_title))
        self.play(FadeIn(tw, shift=UP * 0.1))
        self.play(FadeIn(lw, shift=UP * 0.1))
        self.play(FadeIn(lw_note, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: How it learns — entropy minimization ─────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        how_title = Text(
            "How it learns: entropy minimization",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        how_1 = Text(
            "You don't need labeled data.\n"
            "Just unlabeled test samples.",
            font_size=22, line_spacing=1.4,
        )
        how_2 = Text(
            "The objective: minimize the entropy of the\n"
            "merged model's predictions on unlabeled data.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )

        how_formula = MathTex(
            r"\min_{\{\lambda\}} \; \mathbb{E}_{x} \left[ "
            r"H\!\left( p_{\theta(\lambda)}(y \mid x) \right) \right]",
            font_size=34,
        )

        how_3 = Text(
            "Intuition: a good merge produces confident,\n"
            "low-entropy predictions. A bad merge produces\n"
            "uniform, high-entropy predictions.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )

        how_4 = Text(
            "Optimize the λ's via gradient descent\n"
            "on this unsupervised objective.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )

        how_group = VGroup(how_1, how_2, how_formula, how_3, how_4).arrange(
            DOWN, buff=0.3,
        ).next_to(how_title, DOWN, buff=0.4)

        self.play(Write(how_title))
        self.play(FadeIn(how_1, shift=UP * 0.1))
        self.play(FadeIn(how_2, shift=UP * 0.1))
        self.play(Write(how_formula))
        self.play(FadeIn(how_3, shift=UP * 0.1))
        self.play(FadeIn(how_4, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: Why this matters ─────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        wm_title = Text(
            "Why this matters",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        wm_1 = Text(
            "+11% over Task Arithmetic across 8 tasks.",
            font_size=22, color=GREEN,
        )
        wm_2 = Text(
            "No labeled data or original training data needed.\n"
            "Just forward passes on unlabeled samples.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        wm_3 = Text(
            "Bridges hand-crafted merging and learned merging.",
            font_size=21,
        )
        wm_4 = Text(
            'Conceptually: "what if we just optimize the recipe?"',
            font_size=20, color=GRAY_B, slant=ITALIC,
        )

        wm_group = VGroup(wm_1, wm_2, wm_3, wm_4).arrange(
            DOWN, buff=0.4,
        ).next_to(wm_title, DOWN, buff=0.5)

        self.play(Write(wm_title))
        self.play(FadeIn(wm_1, shift=UP * 0.1))
        self.play(FadeIn(wm_2, shift=UP * 0.1))
        self.play(FadeIn(wm_3, shift=UP * 0.1))
        self.play(FadeIn(wm_4, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.9: Evolutionary Model Merge ─────────────────────────────────────

class EvolutionaryMergeSlide(Slide):
    """Evolutionary Model Merge: automated search over merge recipes."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Evolutionary Model Merge", font_size=44, weight=BOLD)
        paper = Text(
            "Sakana AI / Akiba et al.",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.25)
        venue = Text(
            "Nature Machine Intelligence, 2025",
            font_size=18, color=GRAY_C,
        ).next_to(paper, DOWN, buff=0.15)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.play(FadeIn(venue, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(paper), FadeOut(venue))

        # ── Beat 2: The motivation ───────────────────────────────────────────
        mot_title = Text(
            "The motivation",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        mot_1 = Text(
            "Model merging has been called\n"
            '"a form of black art or alchemy."',
            font_size=22, slant=ITALIC, line_spacing=1.4,
        )
        mot_2 = Text(
            "Every method so far requires humans to choose:\n"
            "which models, which method, which coefficients.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        mot_3 = Text(
            "What if we let an algorithm search\n"
            "over the entire space of possible merges?",
            font_size=22, color=YELLOW, weight=BOLD, line_spacing=1.4,
        )

        mot_group = VGroup(mot_1, mot_2, mot_3).arrange(
            DOWN, buff=0.4,
        ).next_to(mot_title, DOWN, buff=0.5)

        self.play(Write(mot_title))
        self.play(FadeIn(mot_1, shift=UP * 0.1))
        self.play(FadeIn(mot_2, shift=UP * 0.1))
        self.play(FadeIn(mot_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: Two search spaces ────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        ss_title = Text(
            "Two search spaces",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Parameter space
        ps_label = Text(
            "1. Parameter Space (PS)",
            font_size=24, weight=BOLD, color=SAFETY_COLOR,
        )
        ps_desc = Text(
            "Optimize merging coefficients per layer.\n"
            "Uses DARE-TIES as the merge method,\n"
            "but the sparsification rate and weight\n"
            "for each layer are evolved independently.",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )
        ps = VGroup(ps_label, ps_desc).arrange(DOWN, buff=0.12)

        # Data flow space
        dfs_label = Text(
            "2. Data Flow Space (DFS)",
            font_size=24, weight=BOLD, color=CODE_COLOR,
        )
        dfs_desc = Text(
            "Optimize which layers from which models\n"
            "tokens pass through during inference.\n"
            "After layer i of Model A, route to layer j\n"
            "of Model B. A principled frankenmerge.",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )
        dfs = VGroup(dfs_label, dfs_desc).arrange(DOWN, buff=0.12)

        dfs_note = Text(
            "This is what makes it novel — you can create\n"
            "entirely new architectures from existing blocks.",
            font_size=20, color=GREEN, line_spacing=1.3,
        )

        ss_group = VGroup(ps, dfs, dfs_note).arrange(
            DOWN, buff=0.35,
        ).next_to(ss_title, DOWN, buff=0.4)

        self.play(Write(ss_title))
        self.play(FadeIn(ps, shift=UP * 0.1))
        self.play(FadeIn(dfs, shift=UP * 0.1))
        self.play(FadeIn(dfs_note, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: The evolutionary loop ────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        loop_title = Text(
            "The evolutionary loop",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        # Simple text-based pipeline
        step_1 = Text(
            "1. Initialize a population of merge candidates\n"
            "   (random coefficients / layer orderings)",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        step_2 = Text(
            "2. Evaluate each candidate on task-specific\n"
            "   metrics (accuracy for math, ROUGE for VQA)\n"
            "   via forward pass only — no training",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        step_3 = Text(
            "3. Select the fittest candidates",
            font_size=20, color=GRAY_B,
        )
        step_4 = Text(
            "4. Mutate and recombine (CMA-ES)\n"
            "   to produce the next generation",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        step_5 = Text(
            "5. Repeat until convergence",
            font_size=20, color=GRAY_B,
        )

        cost_note = Text(
            "Cost: many forward-pass evaluations.\n"
            "No backward passes. No gradient computation.",
            font_size=19, color=YELLOW, line_spacing=1.3,
        )

        loop_group = VGroup(step_1, step_2, step_3, step_4, step_5, cost_note).arrange(
            DOWN, buff=0.2,
        ).next_to(loop_title, DOWN, buff=0.4)

        self.play(Write(loop_title))
        self.play(FadeIn(step_1, shift=UP * 0.1))
        self.play(FadeIn(step_2, shift=UP * 0.1))
        self.play(FadeIn(step_3, shift=UP * 0.1))
        self.play(FadeIn(step_4, shift=UP * 0.1))
        self.play(FadeIn(step_5, shift=UP * 0.1))
        self.play(FadeIn(cost_note, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: Results ──────────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        res_title = Text(
            "Results",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        res_1 = Text(
            "EvoLLM-JP: a 7B Japanese math LLM\n"
            "that beat previous 70B models on MGSM-JA.",
            font_size=22, color=GREEN, line_spacing=1.4,
        )
        res_2 = Text(
            "Built entirely by merging existing open-source\n"
            "models. No training. No new data.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        res_3 = Text(
            "EvoVLM-JP: a Japanese Vision-Language Model\n"
            "that outperformed previous Japanese VLMs\n"
            "on culture-specific content.",
            font_size=21, color=GRAY_B, line_spacing=1.3,
        )
        res_4 = Text(
            "Cross-domain merging: combined a Japanese\n"
            "language model with an English math model\n"
            "to get Japanese math capability.",
            font_size=21, color=GRAY_B, line_spacing=1.3,
        )

        res_group = VGroup(res_1, res_2, res_3, res_4).arrange(
            DOWN, buff=0.3,
        ).next_to(res_title, DOWN, buff=0.5)

        self.play(Write(res_title))
        self.play(FadeIn(res_1, shift=UP * 0.1))
        self.play(FadeIn(res_2, shift=UP * 0.1))
        self.play(FadeIn(res_3, shift=UP * 0.1))
        self.play(FadeIn(res_4, shift=UP * 0.1))
        self.next_slide()

        self.next_slide()  # end of EvolutionaryMergeSlide


# ─── Slide 6.10: WARM (§6.10) ───────────────────────────────────────────────

class WARMSlide(Slide):
    """WARM: merging reward models for better RLHF alignment."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("WARM", font_size=48, weight=BOLD)
        full_name = Text(
            "Weight Averaged Reward Models",
            font_size=24, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.25)
        cite = Text(
            "Rame et al., ICML 2024, DeepMind",
            font_size=18, color=GRAY_C,
        ).next_to(full_name, DOWN, buff=0.15)

        self.play(Write(title), FadeIn(full_name, shift=UP * 0.1))
        self.play(FadeIn(cite, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(full_name), FadeOut(cite))

        # ── Beat 2: The problem — reward hacking ────────────────────────────
        prob_title = Text(
            "The problem: reward hacking",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        prob_1 = Text(
            "In RLHF, a reward model scores the LLM's\n"
            "outputs to guide training toward human preferences.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        prob_2 = Text(
            "But the LLM learns to exploit flaws in the\n"
            "reward model — achieving high reward scores\n"
            "without actually being better.",
            font_size=21, color=GRAY_B, line_spacing=1.3,
        )
        prob_3 = Text(
            "This is reward hacking.",
            font_size=24, color=RED, weight=BOLD,
        )
        prob_4 = Text(
            "Two root causes:\n"
            "1. Distribution shift during RL training\n"
            "2. Inconsistencies in human preference annotations",
            font_size=19, color=GRAY_C, line_spacing=1.3,
        )

        prob_group = VGroup(prob_1, prob_2, prob_3, prob_4).arrange(
            DOWN, buff=0.3,
        ).next_to(prob_title, DOWN, buff=0.4)

        self.play(Write(prob_title))
        self.play(FadeIn(prob_1, shift=UP * 0.1))
        self.play(FadeIn(prob_2, shift=UP * 0.1))
        self.play(FadeIn(prob_3, shift=UP * 0.1))
        self.play(FadeIn(prob_4, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: The idea ─────────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        idea_title = Text(
            "The idea",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        idea_1 = Text(
            "Train multiple reward models independently\n"
            "(same architecture, different seeds / data splits).",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        idea_2 = Text(
            "Instead of ensembling predictions (expensive),\n"
            "average the weights (free at inference time).",
            font_size=21, line_spacing=1.4,
        )
        idea_formula = MathTex(
            r"\theta_{\text{RM}}^{\text{WARM}} = \frac{1}{N} \sum_{i=1}^{N} \theta_{\text{RM}_i}",
            font_size=36,
        )
        idea_3 = Text(
            "This works because the RMs are fine-tuned\n"
            "from the same pretrained base → they live in\n"
            "the same loss basin (linear mode connectivity).",
            font_size=19, color=GRAY_C, line_spacing=1.3,
        )

        idea_group = VGroup(idea_1, idea_2, idea_formula, idea_3).arrange(
            DOWN, buff=0.3,
        ).next_to(idea_title, DOWN, buff=0.4)

        self.play(Write(idea_title))
        self.play(FadeIn(idea_1, shift=UP * 0.1))
        self.play(FadeIn(idea_2, shift=UP * 0.1))
        self.play(Write(idea_formula))
        self.play(FadeIn(idea_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: Why it works — merging as regularization ─────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        why_title = Text(
            "Why it works: merging as regularization",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        why_1 = Text(
            "Each individual RM overfits to the quirks\n"
            "of its training data — annotation noise,\n"
            "distribution-specific patterns.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        why_2 = Text(
            "Averaging weights smooths out these\n"
            "individual biases. The shared signal survives;\n"
            "the idiosyncratic noise cancels.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        why_3 = Text(
            "The averaged RM is harder to hack because\n"
            "it doesn't have any single exploitable flaw.",
            font_size=21, weight=BOLD, line_spacing=1.4,
        )

        why_group = VGroup(why_1, why_2, why_3).arrange(
            DOWN, buff=0.35,
        ).next_to(why_title, DOWN, buff=0.5)

        self.play(Write(why_title))
        self.play(FadeIn(why_1, shift=UP * 0.1))
        self.play(FadeIn(why_2, shift=UP * 0.1))
        self.play(FadeIn(why_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: Results + significance ───────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        res_title = Text(
            "Results & significance",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        res_1 = Text(
            "79.4% win rate over single-RM baseline\n"
            "on summarization tasks.",
            font_size=22, color=GREEN, line_spacing=1.4,
        )
        res_2 = Text(
            "More robust to distribution shifts than\n"
            "both single RMs and prediction ensembles.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        res_3 = Text(
            "Same cost as a single RM at inference time\n"
            "(unlike ensembles which need N forward passes).",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        res_4 = Text(
            "Merging applied to alignment, not capabilities.",
            font_size=22, color=YELLOW, weight=BOLD,
        )

        res_group = VGroup(res_1, res_2, res_3, res_4).arrange(
            DOWN, buff=0.35,
        ).next_to(res_title, DOWN, buff=0.5)

        self.play(Write(res_title))
        self.play(FadeIn(res_1, shift=UP * 0.1))
        self.play(FadeIn(res_2, shift=UP * 0.1))
        self.play(FadeIn(res_3, shift=UP * 0.1))
        self.play(FadeIn(res_4, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.11: Training for Mergeability (§6.11) ──────────────────────────

class MergeabilitySlide(Slide):
    """Training for mergeability via effective noise scale."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Training for Mergeability", font_size=44, weight=BOLD)
        cite = Text(
            "Zhang et al., 2025",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.25)
        paper_name = Text(
            '"How does the optimizer implicitly bias\n'
            'the model merging loss landscape?"',
            font_size=18, color=GRAY_C, slant=ITALIC, line_spacing=1.3,
        ).next_to(cite, DOWN, buff=0.15)

        self.play(Write(title), FadeIn(cite, shift=UP * 0.1))
        self.play(FadeIn(paper_name, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(cite), FadeOut(paper_name))

        # ── Beat 2: The puzzle ───────────────────────────────────────────────
        puz_title = Text(
            "The puzzle",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        puz_1 = Text(
            "Two models trained on the same task,\n"
            "same architecture, similar final accuracy.",
            font_size=22, line_spacing=1.4,
        )
        puz_2 = Text(
            "One pair merges well. The other doesn't.",
            font_size=22, color=YELLOW, weight=BOLD,
        )
        puz_3 = Text(
            "The only difference: training hyperparameters\n"
            "(learning rate, batch size, weight decay).",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        puz_4 = Text(
            "The paper asks: what property of the\n"
            "optimizer determines merging compatibility?",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )

        puz_group = VGroup(puz_1, puz_2, puz_3, puz_4).arrange(
            DOWN, buff=0.35,
        ).next_to(puz_title, DOWN, buff=0.4)

        self.play(Write(puz_title))
        self.play(FadeIn(puz_1, shift=UP * 0.1))
        self.play(FadeIn(puz_2, shift=UP * 0.1))
        self.play(FadeIn(puz_3, shift=UP * 0.1))
        self.play(FadeIn(puz_4, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: The key insight — optimization noise ─────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        noise_title = Text(
            "The answer: optimization noise",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        noise_1 = Text(
            "SGD doesn't follow the true gradient exactly.\n"
            "Each mini-batch gives a noisy estimate.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        noise_2 = Text(
            "The paper shows that the magnitude of this\n"
            "noise during training determines where in the\n"
            "loss landscape the model ends up — and whether\n"
            "that region is compatible with other models.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        noise_3 = Text(
            "They capture this with a single number:",
            font_size=21,
        )
        noise_formula = MathTex(
            r"\tilde{\mathcal{S}} = \frac{\eta}{B \, (1 - \mu)^2}",
            font_size=38,
        )
        noise_gloss = Text(
            "η = learning rate,  B = batch size,  μ = momentum",
            font_size=17, color=GRAY_C,
        )

        noise_group = VGroup(noise_1, noise_2, noise_3, noise_formula, noise_gloss).arrange(
            DOWN, buff=0.25,
        ).next_to(noise_title, DOWN, buff=0.4)

        self.play(Write(noise_title))
        self.play(FadeIn(noise_1, shift=UP * 0.1))
        self.play(FadeIn(noise_2, shift=UP * 0.1))
        self.play(FadeIn(noise_3, shift=UP * 0.1))
        self.play(Write(noise_formula))
        self.play(FadeIn(noise_gloss, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: The non-monotonic finding ────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        nm_title = Text(
            "The finding: a non-monotonic relationship",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        nm_desc = Text(
            "The paper finds that merging effectiveness\n"
            "is a non-monotonic function of noise scale.\n"
            "There is a distinct optimum.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        ).next_to(nm_title, DOWN, buff=0.3)

        # Inverted-U curve
        ax = Axes(
            x_range=[0, 5, 1], y_range=[0, 1.2, 0.5],
            x_length=7, y_length=2.8,
            axis_config={"include_numbers": False, "stroke_opacity": 0.4},
            tips=False,
        ).shift(DOWN * 0.8)

        curve = ax.plot(
            lambda x: 1.0 * np.exp(-((x - 2.5)**2) / 1.2),
            x_range=[0.2, 4.8], color=GREEN, stroke_width=3,
        )

        low_lbl = Text("Low noise:\nnarrow minima,\nhard to merge", font_size=13, color=RED, line_spacing=1.2)
        low_lbl.next_to(ax.c2p(0.7, 0.5), RIGHT, buff=0.1)
        mid_lbl = Text("Optimal noise", font_size=14, color=GREEN, weight=BOLD)
        mid_lbl.next_to(ax.c2p(2.5, 1.05), UP, buff=0.1)
        high_lbl = Text("High noise:\nunstable training", font_size=13, color=RED, line_spacing=1.2)
        high_lbl.next_to(ax.c2p(4.3, 0.5), LEFT, buff=0.1)

        x_lbl = Text("Effective Noise Scale", font_size=15, color=GRAY_C).next_to(ax, DOWN, buff=0.3)
        y_lbl = Text("Accuracy gain\nfrom merging", font_size=13, color=GRAY_C, line_spacing=1.2)
        y_lbl.next_to(ax, LEFT, buff=0.2)

        self.play(Write(nm_title), FadeIn(nm_desc, shift=UP * 0.1))
        self.play(
            Create(ax), Create(curve),
            FadeIn(x_lbl), FadeIn(y_lbl),
        )
        self.play(FadeIn(low_lbl), FadeIn(mid_lbl), FadeIn(high_lbl))
        self.next_slide()

        # ── Beat 5: Why does noise help? ─────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        why_title = Text(
            "Why does noise help?",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        why_1 = Text(
            "SGD with more noise is unstable in sharp minima —\n"
            "it bounces out of them. This biases training toward\n"
            "wider, flatter regions. (Keskar et al. 2016)",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        )
        why_2 = Text(
            "The paper's new observation:",
            font_size=21, weight=BOLD,
        )
        why_3 = Text(
            "When two independent training runs both have\n"
            "high noise, they're both biased toward the same\n"
            "wide region. Their solutions overlap more.",
            font_size=20, line_spacing=1.3,
        )
        why_4 = Text(
            "Low noise → nearby but distinct narrow sub-valleys\n"
            "within the basin. Interpolation crosses small ridges.",
            font_size=19, color=RED, line_spacing=1.3,
        )
        why_5 = Text(
            "High noise → wide, overlapping sub-valleys.\n"
            "Interpolation stays flat.",
            font_size=19, color=GREEN, line_spacing=1.3,
        )

        why_group = VGroup(why_1, why_2, why_3, why_4, why_5).arrange(
            DOWN, buff=0.25,
        ).next_to(why_title, DOWN, buff=0.4)

        self.play(Write(why_title))
        self.play(FadeIn(why_1, shift=UP * 0.1))
        self.play(FadeIn(why_2, shift=UP * 0.1))
        self.play(FadeIn(why_3, shift=UP * 0.1))
        self.play(FadeIn(why_4, shift=UP * 0.1))
        self.play(FadeIn(why_5, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 6: Paper figure — empirical evidence ────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        fig_title = Text(
            "Empirical evidence (ResNet-18 on CIFAR-100)",
            font_size=24, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        fig_img = ImageMobject("assets/noise_scale_curves.png")
        fig_img.scale_to_fit_width(11)
        fig_img.next_to(fig_title, DOWN, buff=0.25)

        fig_caption = Text(
            "When reparameterized as effective noise (c), different\n"
            "LR/batch-size curves collapse onto a single non-monotonic curve.",
            font_size=17, color=GRAY_C, line_spacing=1.3,
        ).next_to(fig_img, DOWN, buff=0.2)

        self.play(Write(fig_title))
        self.play(FadeIn(fig_img), FadeIn(fig_caption))
        self.next_slide()

        # ── Beat 7: What each hyperparameter does ────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        hp_title = Text(
            "Each hyperparameter modulates the noise",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        hp_desc = Text(
            "The paper decomposes the effective noise scale\n"
            "and shows each component independently affects\n"
            "merging effectiveness in the same qualitative way:",
            font_size=19, color=GRAY_B, line_spacing=1.3,
        ).next_to(hp_title, DOWN, buff=0.3)

        hp_items = Text(
            "↑ Learning rate    → ↑ noise → more compatible merges\n"
            "↑ Weight decay    → ↑ effective LR → same effect\n"
            "↓ Batch size         → noisier gradients → same effect\n"
            "↑ Augmentation   → noisier data → same effect",
            font_size=19, color=GREEN, line_spacing=1.5,
        ).next_to(hp_desc, DOWN, buff=0.3)

        hp_caveat = Text(
            "Key: models trained with similar noise scales\n"
            "are more compatible with each other for merging.\n"
            "Confirmed across MLPs, ResNets, DenseNets,\n"
            "Transformers, and GPT on vision + language tasks.",
            font_size=18, color=GRAY_C, line_spacing=1.3,
        ).next_to(hp_items, DOWN, buff=0.3)

        self.play(Write(hp_title))
        self.play(FadeIn(hp_desc, shift=UP * 0.1))
        self.play(FadeIn(hp_items, shift=UP * 0.1))
        self.play(FadeIn(hp_caveat, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.12: Frankenmerges (§6.12) ──────────────────────────────────────

class FrankenmergesSlide(Slide):
    """Frankenmerges / Passthrough Merging: community-driven layer stacking."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Frankenmerges", font_size=48, weight=BOLD)
        aka = Text(
            "a.k.a. Passthrough Merging",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.25)
        cite = Text(
            "Community-driven, 2023–2024",
            font_size=18, color=GRAY_C,
        ).next_to(aka, DOWN, buff=0.15)

        self.play(Write(title), FadeIn(aka, shift=UP * 0.1))
        self.play(FadeIn(cite, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(aka), FadeOut(cite))

        # ── Beat 2: What it is ───────────────────────────────────────────────
        what_title = Text(
            "What is frankenmerging?",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        what_1 = Text(
            "Not a paper. A technique developed by the\n"
            "open-source model merging community.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        what_2 = Text(
            "Instead of blending weights, stack layers\n"
            "from different models sequentially:",
            font_size=21, line_spacing=1.4,
        )
        what_example = Text(
            "Layers 0–16 from Model A\n"
            "Layers 17–32 from Model B",
            font_size=22, color=SAFETY_COLOR, line_spacing=1.4,
        )
        what_3 = Text(
            "The resulting model is a Frankenstein's monster\n"
            "of parts from different fine-tunes.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        what_group = VGroup(what_1, what_2, what_example, what_3).arrange(
            DOWN, buff=0.3,
        ).next_to(what_title, DOWN, buff=0.4)

        self.play(Write(what_title))
        self.play(FadeIn(what_1, shift=UP * 0.1))
        self.play(FadeIn(what_2, shift=UP * 0.1))
        self.play(FadeIn(what_example, shift=UP * 0.1))
        self.play(FadeIn(what_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: Upscaling — more layers than parents ─────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        up_title = Text(
            "You can even upscale",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        up_1 = Text(
            "Select subsets of layers from multiple models\n"
            "to create a model with more layers than either parent.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        up_example = Text(
            "Two 32-layer models → a 48-layer model\n"
            "(picking 24 layers from each)",
            font_size=22, color=CODE_COLOR, line_spacing=1.4,
        )
        up_2 = Text(
            "This is a form of model scaling that requires\n"
            "zero training — just careful layer selection.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        up_group = VGroup(up_1, up_example, up_2).arrange(
            DOWN, buff=0.35,
        ).next_to(up_title, DOWN, buff=0.5)

        self.play(Write(up_title))
        self.play(FadeIn(up_1, shift=UP * 0.1))
        self.play(FadeIn(up_example, shift=UP * 0.1))
        self.play(FadeIn(up_2, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: No theory, but results ───────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        nt_title = Text(
            "No theory. But results.",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        nt_1 = Text(
            "No theoretical justification for why this works.",
            font_size=21, color=GRAY_B,
        )
        nt_2 = Text(
            "But frankenmerges produced several of the\n"
            "top models on the Open LLM Leaderboard\n"
            "throughout 2023–2024.",
            font_size=21, color=GREEN, line_spacing=1.3,
        )
        nt_3 = Text(
            "Everyone used roughly the same recipe —\n"
            "the method space is highly under-explored.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )
        nt_4 = Text(
            "The community was ahead of academia\n"
            "in exploring what was possible with merging.",
            font_size=20, color=GRAY_C, line_spacing=1.4,
        )

        nt_group = VGroup(nt_1, nt_2, nt_3, nt_4).arrange(
            DOWN, buff=0.35,
        ).next_to(nt_title, DOWN, buff=0.5)

        self.play(Write(nt_title))
        self.play(FadeIn(nt_1, shift=UP * 0.1))
        self.play(FadeIn(nt_2, shift=UP * 0.1))
        self.play(FadeIn(nt_3, shift=UP * 0.1))
        self.play(FadeIn(nt_4, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 6.5: DARE (§6.7) ─────────────────────────────────────────────────

class DARESlide(Slide):
    """DARE: Drop And REscale — sparsify deltas before merging."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("DARE: Drop And REscale", font_size=44, weight=BOLD)
        paper = Text(
            "Yu et al., ICML 2024  (~300 citations)",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.25)
        nickname = Text(
            '"Language Models are Super Mario"',
            font_size=18, color=GRAY_C, slant=ITALIC,
        ).next_to(paper, DOWN, buff=0.15)

        self.play(Write(title), FadeIn(paper, shift=UP * 0.1))
        self.play(FadeIn(nickname, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(paper), FadeOut(nickname))

        # ── Beat 2: The key observation ──────────────────────────────────────
        obs_title = Text(
            "The observation",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        obs_1 = Text(
            "Look at the delta parameters after\n"
            "supervised fine-tuning:",
            font_size=22, line_spacing=1.4,
        )

        obs_mag = MathTex(
            r"\tau_j = \theta^{\text{ft}}_j - \theta^{\text{pre}}_j"
            r"\quad \approx \quad 0.002 \text{ -- } 0.005",
            font_size=30,
        )

        obs_2 = Text(
            "Their magnitudes are tiny.",
            font_size=22, color=YELLOW, weight=BOLD,
        )
        obs_3 = Text(
            "Most of them are noise.\n"
            "Extreme redundancy.",
            font_size=22, color=GRAY_B, line_spacing=1.4,
        )

        obs_group = VGroup(obs_1, obs_mag, obs_2, obs_3).arrange(
            DOWN, buff=0.4,
        ).next_to(obs_title, DOWN, buff=0.5)

        self.play(Write(obs_title))
        self.play(FadeIn(obs_1, shift=UP * 0.1))
        self.play(Write(obs_mag))
        self.play(FadeIn(obs_2, shift=UP * 0.1))
        self.play(FadeIn(obs_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: The idea — two steps ─────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        idea_title = Text(
            "The idea: what if we just threw most of them away?",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        step1_label = Text(
            "Step 1: Drop", font_size=24, weight=BOLD, color=RED,
        )
        step1_desc = Text(
            "Randomly set delta parameters to zero\n"
            "with probability p.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        step1_math = MathTex(
            r"\tilde{\tau}_j = \begin{cases} 0 & \text{with prob } p \\"
            r" \tau_j & \text{with prob } 1-p \end{cases}",
            font_size=28,
        )

        step2_label = Text(
            "Step 2: Rescale", font_size=24, weight=BOLD, color=GREEN,
        )
        step2_desc = Text(
            "Multiply survivors by 1/(1−p)\n"
            "to preserve the expected sum.",
            font_size=20, color=GRAY_B, line_spacing=1.3,
        )
        step2_math = MathTex(
            r"\hat{\tau}_j = \frac{\tilde{\tau}_j}{1 - p}",
            font_size=32,
        )

        step1 = VGroup(step1_label, step1_desc, step1_math).arrange(
            DOWN, buff=0.15,
        )
        step2 = VGroup(step2_label, step2_desc, step2_math).arrange(
            DOWN, buff=0.15,
        )
        steps = VGroup(step1, step2).arrange(
            DOWN, buff=0.5,
        ).next_to(idea_title, DOWN, buff=0.4)

        self.play(Write(idea_title))
        self.play(FadeIn(step1, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeIn(step2, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: Why the rescale? ─────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        resc_title = Text(
            "Why rescale by 1/(1−p)?",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        resc_1 = Text(
            "If you drop p fraction of values,\n"
            "the expected sum shrinks by factor (1−p).",
            font_size=22, line_spacing=1.4,
        )

        resc_math = MathTex(
            r"\mathbb{E}[\tilde{\tau}_j] = (1-p) \cdot \tau_j",
            font_size=32,
        )

        resc_2 = Text(
            "Rescaling compensates exactly:",
            font_size=22,
        )

        resc_math2 = MathTex(
            r"\mathbb{E}\!\left[\frac{\tilde{\tau}_j}{1-p}\right] = \tau_j",
            font_size=32,
        )

        resc_3 = Text(
            "Same principle as dropout during training —\n"
            "same math, applied post-hoc to the delta.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        resc_group = VGroup(resc_1, resc_math, resc_2, resc_math2, resc_3).arrange(
            DOWN, buff=0.35,
        ).next_to(resc_title, DOWN, buff=0.5)

        self.play(Write(resc_title))
        self.play(FadeIn(resc_1, shift=UP * 0.1))
        self.play(Write(resc_math))
        self.play(FadeIn(resc_2, shift=UP * 0.1))
        self.play(Write(resc_math2))
        self.play(FadeIn(resc_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: Visual — the sparsification ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        vis_title = Text(
            "Sparsifying a delta vector",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        # Generate delta values: mostly tiny, a few larger
        np.random.seed(42)
        raw_deltas = np.random.randn(16) * 0.003
        raw_deltas[3] = 0.02
        raw_deltas[7] = -0.018
        raw_deltas[11] = 0.015

        bar_w = 0.35
        max_h = 2.5
        scale_factor = max_h / 0.025  # normalize so 0.025 = full height

        def make_bars(values, color, y_center, opacity=0.8):
            bars = VGroup()
            for i, v in enumerate(values):
                h = abs(v) * scale_factor
                h = max(h, 0.03)
                direction = 1 if v >= 0 else -1
                bar = Rectangle(
                    width=bar_w, height=h,
                    fill_color=color, fill_opacity=opacity,
                    stroke_width=0.5, stroke_color=WHITE,
                )
                bar.move_to([
                    -len(values) * bar_w * 0.7 / 2 + i * bar_w * 0.7,
                    y_center + direction * h / 2,
                    0,
                ])
                bars.add(bar)
            return bars

        # Original deltas
        original_label = Text(
            "Original delta (τ)", font_size=18, color=SAFETY_COLOR,
        ).shift(LEFT * 5 + UP * 0.5)
        original_bars = make_bars(raw_deltas, SAFETY_COLOR, y_center=0.5)

        self.play(Write(vis_title))
        self.play(FadeIn(original_label), FadeIn(original_bars))
        self.next_slide()

        # Drop: randomly zero out most (keep indices 3, 7, 11 and a couple others)
        keep_mask = np.zeros(16, dtype=bool)
        keep_mask[[3, 7, 11]] = True  # signal parameters survive
        dropped_deltas = raw_deltas.copy()
        dropped_deltas[~keep_mask] = 0

        dropped_bars = make_bars(dropped_deltas, RED, y_center=0.5, opacity=0.5)

        drop_label = Text(
            "After drop (p ≈ 0.8)", font_size=18, color=RED,
        ).shift(LEFT * 5 + DOWN * 0.3)

        # Animate: original bars fade, dropped bars appear
        self.play(
            original_bars.animate.set_opacity(0.15),
            FadeIn(dropped_bars),
            FadeIn(drop_label),
        )
        self.next_slide()

        # Rescale: survivors get taller
        p_drop = 1 - keep_mask.sum() / len(keep_mask)
        rescaled_deltas = dropped_deltas / (1 - p_drop)
        rescaled_bars = make_bars(rescaled_deltas, GREEN, y_center=0.5)

        rescale_label = Text(
            "After rescale ×1/(1−p)", font_size=18, color=GREEN,
        ).shift(LEFT * 5 + DOWN * 1.1)

        self.play(
            FadeOut(dropped_bars),
            FadeIn(rescaled_bars),
            FadeIn(rescale_label),
        )

        surviving_note = Text(
            "The signal survives. The noise is gone.",
            font_size=22, weight=BOLD,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(surviving_note, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 6: Scale dependence ─────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        scale_title = Text(
            "Larger models tolerate higher drop rates",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        scale_data = [
            ("7B", "~90%", SAFETY_COLOR),
            ("13B", "~95%", CODE_COLOR),
            ("70B", "~99%", RLVR_COLOR),
        ]

        scale_rows = VGroup()
        for size, rate, color in scale_data:
            size_t = Text(size, font_size=28, weight=BOLD, color=color)
            arrow = Text("→", font_size=24, color=GRAY_C)
            rate_t = Text(f"drop {rate}", font_size=24)
            row = VGroup(size_t, arrow, rate_t).arrange(RIGHT, buff=0.4)
            scale_rows.add(row)

        scale_rows.arrange(DOWN, buff=0.5).next_to(scale_title, DOWN, buff=0.6)

        scale_why = Text(
            "Larger models have more redundancy\n"
            "in their delta parameters.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        ).next_to(scale_rows, DOWN, buff=0.5)

        self.play(Write(scale_title))
        for row in scale_rows:
            self.play(FadeIn(row, shift=UP * 0.1), run_time=0.7)
        self.play(FadeIn(scale_why, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 7: DARE-TIES & practical impact ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        combo_title = Text(
            "DARE-TIES",
            font_size=32, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        combo_1 = Text(
            "Combine both methods:",
            font_size=22,
        )
        combo_pipeline = Text(
            "1. Sparsify deltas (DARE)\n"
            "2. Resolve sign conflicts (TIES)\n"
            "3. Merge",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        combo_impact = Text(
            "The most popular practical method in 2024–25.\n"
            "Produced #1 models on the Open LLM Leaderboard.",
            font_size=21, line_spacing=1.4,
        )
        combo_della = Text(
            "DELLA (2024): magnitude-based dropout\n"
            "instead of uniform random.\n"
            '"Drop the small ones first."',
            font_size=19, color=GRAY_C, line_spacing=1.3,
        )

        combo_group = VGroup(combo_1, combo_pipeline, combo_impact, combo_della).arrange(
            DOWN, buff=0.4,
        ).next_to(combo_title, DOWN, buff=0.5)

        self.play(Write(combo_title))
        self.play(FadeIn(combo_1, shift=UP * 0.1))
        self.play(FadeIn(combo_pipeline, shift=UP * 0.1))
        self.play(FadeIn(combo_impact, shift=UP * 0.1))
        self.play(FadeIn(combo_della, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 8: The counterintuitive takeaway ────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        takeaway = Text(
            "Throwing away almost everything\n"
            "works better than keeping it all.",
            font_size=34, weight=BOLD, line_spacing=1.4,
        )

        self.play(FadeIn(takeaway, scale=1.1), run_time=1.0)
        self.next_slide()


# ─── Slide 8+9: Practical Implications ───────────────────────────────────────

class PracticalImplicationsSlide(Slide):
    """What matters in practice — 5 key findings from Yadav et al. 2024."""

    def construct(self):
        cite = Text(
            'Yadav et al., "What Matters for Model Merging at Scale?", 2024',
            font_size=14, color=GRAY_C,
        ).to_corner(DR, buff=0.25)

        # ── Beat 1: Base model quality ───────────────────────────────────────
        title = Text("Practical Implications", font_size=44, weight=BOLD)
        subtitle = Text(
            "What actually matters when merging models",
            font_size=22, color=GRAY_B,
        ).next_to(title, DOWN, buff=0.3)

        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(subtitle), FadeIn(cite))

        pt1_title = Text(
            "1. Base model quality is the single biggest factor",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        pt1_text = Text(
            "Merging works dramatically better when experts are\n"
            "fine-tuned from strong base models with good\n"
            "zero-shot performance.\n\n"
            "Start from the best base you can get.\n"
            "This matters more than your choice of merging method.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        ).next_to(pt1_title, DOWN, buff=0.5)

        self.play(Write(pt1_title))
        self.play(FadeIn(pt1_text, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 2: Larger models merge more easily ──────────────────────────
        self.play(FadeOut(pt1_title), FadeOut(pt1_text))

        pt2_title = Text(
            "2. Larger models merge more easily",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        pt2_img = ImageMobject("assets/bigger_models_better.png")
        pt2_img.scale_to_fit_width(11)
        pt2_img.next_to(pt2_title, DOWN, buff=0.2)

        pt2_text = Text(
            "At 1B, merging is fragile. At 64B, it's reliable.\n"
            "Larger models have flatter loss basins,\n"
            "task vectors are more orthogonal,\n"
            "and fine-tuning perturbs less relative to the base.",
            font_size=18, color=GRAY_B, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(pt2_title))
        self.play(FadeIn(pt2_img))
        self.play(FadeIn(pt2_text, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: Merging improves generalization ──────────────────────────
        self.play(FadeOut(pt2_title), FadeOut(pt2_img), FadeOut(pt2_text))

        pt3_title = Text(
            "3. Merging improves generalization",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        pt3_img = ImageMobject("assets/heldout_performance.png")
        pt3_img.scale_to_fit_width(11)
        pt3_img.next_to(pt3_title, DOWN, buff=0.2)

        pt3_text = Text(
            "Merged models generalize better to held-out tasks\n"
            "than individual experts — and at scale, can outperform\n"
            "multitask-trained models.",
            font_size=18, color=GRAY_B, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(pt3_title))
        self.play(FadeIn(pt3_img))
        self.play(FadeIn(pt3_text, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3b: Nuanced take — not always ──────────────────────────────
        self.play(FadeOut(pt3_img), FadeOut(pt3_text))

        pt3b_title = Text(
            "...but not always",
            font_size=26, weight=BOLD, color=YELLOW,
        ).next_to(pt3_title, DOWN, buff=0.3)

        pt3b_img = ImageMobject("assets/generalization_nuanced.png")
        pt3b_img.scale_to_fit_width(11)
        pt3b_img.next_to(pt3b_title, DOWN, buff=0.2)

        pt3b_text = Text(
            "Merging improves generalization in some domains\n"
            "but not all. The gains depend on task similarity\n"
            "and how well the base model covers the domain.",
            font_size=18, color=GRAY_B, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.4)

        self.play(FadeIn(pt3b_title, shift=UP * 0.1))
        self.play(FadeIn(pt3b_img))
        self.play(FadeIn(pt3b_text, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: Instruction-tuned bases facilitate better merging ────────
        self.play(
            FadeOut(pt3_title), FadeOut(pt3b_title),
            FadeOut(pt3b_img), FadeOut(pt3b_text),
        )

        pt4_title = Text(
            "4. Instruction-tuned models facilitate better merging",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        pt4_img = ImageMobject("assets/it_vs_base_held_in.png")
        pt4_img.scale_to_fit_width(11)
        pt4_img.next_to(pt4_title, DOWN, buff=0.2)

        pt4_text = Text(
            "Instruction-tuned base, 8 experts: performance preserved.\n"
            "Pretrained base, 8 experts: performance destroyed.\n"
            "IT models sit in wider, more compatible basins.",
            font_size=18, color=GRAY_B, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(pt4_title))
        self.play(FadeIn(pt4_img))
        self.play(FadeIn(pt4_text, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: All methods converge at scale ────────────────────────────
        self.play(FadeOut(pt4_title), FadeOut(pt4_img), FadeOut(pt4_text))

        pt5_title = Text(
            "5. All merging methods converge at scale",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.4)

        pt5_img = ImageMobject("assets/merging_methods_similar.png")
        pt5_img.scale_to_fit_width(11)
        pt5_img.next_to(pt5_title, DOWN, buff=0.2)

        pt5_text = Text(
            "At 64B, Averaging, Task Arithmetic, DARE-TIES,\n"
            "and TIES all perform similarly.\n"
            "The landscape is flat enough that the tricks matter less.",
            font_size=18, color=GRAY_B, line_spacing=1.3,
        ).to_edge(DOWN, buff=0.4)

        self.play(Write(pt5_title))
        self.play(FadeIn(pt5_img))
        self.play(FadeIn(pt5_text, shift=UP * 0.1))
        self.next_slide()


# ─── Slide 10: Community & Industry Adoption ─────────────────────────────────

class AdoptionSlide(Slide):
    """Community adoption, SLERP, frontier lab usage, use cases."""

    def construct(self):
        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("Community & Industry Adoption", font_size=44, weight=BOLD)
        self.play(Write(title))
        self.next_slide()
        self.play(FadeOut(title))

        # ── Beat 2: The HuggingFace explosion ────────────────────────────────
        hf_title = Text(
            "The open-source explosion", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        hf_1 = Text(
            "Hundreds of thousands of fine-tuned models on\n"
            "HuggingFace, most derived from the same bases:\n"
            "Llama, Qwen, Mistral.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )
        hf_2 = Text(
            "A massive supply of specialist models\n"
            "sitting in the same loss basin.",
            font_size=22, line_spacing=1.4,
        )
        hf_3 = Text(
            "Model merging was the first technique the\n"
            "open-source community adopted at scale.\n"
            "Merged models dominated the Open LLM\n"
            "Leaderboard throughout 2023-2024.",
            font_size=21, color=GRAY_B, line_spacing=1.4,
        )

        hf_group = VGroup(hf_1, hf_2, hf_3).arrange(
            DOWN, buff=0.4,
        ).next_to(hf_title, DOWN, buff=0.5)

        self.play(Write(hf_title))
        self.play(FadeIn(hf_1, shift=UP * 0.1))
        self.play(FadeIn(hf_2, shift=UP * 0.1))
        self.play(FadeIn(hf_3, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 3: SLERP — the community's go-to ───────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        slerp_title = Text(
            "SLERP: the community's go-to method",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        slerp_desc = Text(
            "Spherical Linear Interpolation — borrowed from\n"
            "computer graphics (quaternion rotation).\n\n"
            "Instead of interpolating along a straight line\n"
            "(which shrinks weight norms at the midpoint),\n"
            "interpolate along the surface of a hypersphere\n"
            "(preserving norms).",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        ).next_to(slerp_title, DOWN, buff=0.4)

        self.play(Write(slerp_title))
        self.play(FadeIn(slerp_desc, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 4: SLERP animation — LERP vs arc ───────────────────────────
        self.play(FadeOut(slerp_desc))

        # Unit circle
        circle = Circle(radius=2.2, color=GRAY_D, stroke_width=1.5)
        circle.shift(DOWN * 0.3)
        center = circle.get_center()

        # Two model points on the circle
        angle_a = 25 * DEGREES
        angle_b = 75 * DEGREES
        pt_a = center + 2.2 * np.array([np.cos(angle_a), np.sin(angle_a), 0])
        pt_b = center + 2.2 * np.array([np.cos(angle_b), np.sin(angle_b), 0])

        dot_a = Dot(pt_a, color=SAFETY_COLOR, radius=0.1)
        dot_b = Dot(pt_b, color=CODE_COLOR, radius=0.1)
        label_a = MathTex(r"\theta_A", font_size=26, color=SAFETY_COLOR).next_to(dot_a, RIGHT, buff=0.15)
        label_b = MathTex(r"\theta_B", font_size=26, color=CODE_COLOR).next_to(dot_b, UP, buff=0.15)

        # LERP: straight chord
        lerp_line = Line(pt_a, pt_b, color=RED, stroke_width=3)
        lerp_mid = (pt_a + pt_b) / 2
        lerp_dot = Dot(lerp_mid, color=RED, radius=0.08)
        lerp_label = Text("LERP", font_size=18, color=RED).next_to(lerp_line, DOWN, buff=0.15)

        # Dashed line from center to LERP midpoint to show shorter norm
        lerp_radius = Line(center, lerp_mid, color=RED, stroke_width=1.5, stroke_opacity=0.5)
        lerp_norm = Text(
            "shorter norm!", font_size=16, color=RED,
        ).next_to(lerp_radius.get_center(), LEFT, buff=0.15)

        # SLERP: arc along the circle
        slerp_arc = Arc(
            radius=2.2,
            start_angle=angle_a, angle=angle_b - angle_a,
            color=GREEN, stroke_width=4,
            arc_center=center,
        )
        slerp_mid_angle = (angle_a + angle_b) / 2
        slerp_mid = center + 2.2 * np.array([np.cos(slerp_mid_angle), np.sin(slerp_mid_angle), 0])
        slerp_dot = Dot(slerp_mid, color=GREEN, radius=0.08)
        slerp_label = Text("SLERP", font_size=18, color=GREEN).next_to(slerp_arc, RIGHT, buff=0.2)

        # Dashed line from center to SLERP midpoint — same length
        slerp_radius = Line(center, slerp_mid, color=GREEN, stroke_width=1.5, stroke_opacity=0.5)
        slerp_norm = Text(
            "same norm", font_size=16, color=GREEN,
        ).next_to(slerp_radius.get_center(), RIGHT, buff=0.15)

        # Origin dot
        origin_dot = Dot(center, color=WHITE, radius=0.05)
        origin_label = Text("origin", font_size=14, color=GRAY_C).next_to(origin_dot, DOWN, buff=0.1)

        self.play(Create(circle), FadeIn(origin_dot), FadeIn(origin_label))
        self.play(FadeIn(dot_a), FadeIn(label_a), FadeIn(dot_b), FadeIn(label_b))

        # Show LERP
        self.play(Create(lerp_line), FadeIn(lerp_label))
        self.play(FadeIn(lerp_dot), Create(lerp_radius), FadeIn(lerp_norm))
        self.next_slide()

        # Show SLERP
        self.play(Create(slerp_arc), FadeIn(slerp_label))
        self.play(FadeIn(slerp_dot), Create(slerp_radius), FadeIn(slerp_norm))

        # Practical note at bottom
        slerp_note = Text(
            "Only works for 2 models. Produced #1 Open LLM Leaderboard model.",
            font_size=18, color=GRAY_C,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(slerp_note, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 5: Frontier labs — overview ─────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        labs_title = Text(
            "Frontier labs use merging", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        labs = VGroup(
            Text("Cohere", font_size=22, weight=BOLD, color=SAFETY_COLOR),
            Text("Recycling suboptimal training checkpoints via\n"
                 "weight merging in ~100B pipeline (2024)",
                 font_size=18, color=GRAY_B, line_spacing=1.3),
            Text("Meta", font_size=22, weight=BOLD, color=CODE_COLOR),
            Text("Branch-Train-MiX: a scalable training\n"
                 "paradigm built on merging",
                 font_size=18, color=GRAY_B, line_spacing=1.3),
            Text("DeepSeek", font_size=22, weight=BOLD, color=RLVR_COLOR),
            Text("Merging during distillation — distill multiple\n"
                 "specialists from teacher, then merge",
                 font_size=18, color=GRAY_B, line_spacing=1.3),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(labs_title, DOWN, buff=0.5).shift(LEFT * 1.5)

        self.play(Write(labs_title))
        for item in labs:
            self.play(FadeIn(item, shift=UP * 0.1), run_time=0.5)
        self.next_slide()

        # ── Beat 6: ByteDance PMA deep dive ──────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        pma_title = Text(
            "ByteDance PMA: merging during pre-training",
            font_size=26, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        pma_cite = Text(
            "Pre-trained Model Average (ByteDance Seed, 2025)",
            font_size=18, color=GRAY_C,
        ).next_to(pma_title, DOWN, buff=0.2)

        pma_what = Text(
            "Merge checkpoints trained with constant learning\n"
            "rates during the stable phase of pre-training.\n"
            "Not post-hoc — merging is part of the training loop.",
            font_size=20, color=GRAY_B, line_spacing=1.4,
        )

        pma_scale = Text(
            "Validated across:\n"
            "  Dense models: 411M → 70B parameters\n"
            "  MoE models: up to 20B/200B (active/total)",
            font_size=20, line_spacing=1.3,
        )

        pma_results = Text(
            "70B model results:\n"
            "  HumanEval:  50.6 → 57.9\n"
            "  GSM8K:      85.9 → 91.3",
            font_size=22, weight=BOLD, color=GREEN, line_spacing=1.3,
        )

        pma_shift = Text(
            "Shifts the paradigm from 'how to merge after training'\n"
            "to 'merging as a first-class training primitive.'",
            font_size=20, color=YELLOW, line_spacing=1.4,
        )

        pma_group = VGroup(pma_what, pma_scale, pma_results, pma_shift).arrange(
            DOWN, buff=0.35,
        ).next_to(pma_cite, DOWN, buff=0.4)

        self.play(Write(pma_title), FadeIn(pma_cite, shift=UP * 0.1))
        self.play(FadeIn(pma_what, shift=UP * 0.1))
        self.play(FadeIn(pma_scale, shift=UP * 0.1))
        self.next_slide()
        self.play(FadeIn(pma_results, shift=UP * 0.1))
        self.play(FadeIn(pma_shift, shift=UP * 0.1))
        self.next_slide()

        # ── Beat 7: Practical use cases ──────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        use_title = Text(
            "Use cases", font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.5)

        uses = [
            ("Combining capabilities cheaply",
             "Merge code + chat + instruction specialists.\nNo GPU needed — just weight arithmetic.", SAFETY_COLOR),
            ("Decentralized training",
             "Teams fine-tune independently, combine results\nwithout sharing proprietary data.", CODE_COLOR),
            ("Continual learning",
             "Merge in new knowledge without catastrophic\nforgetting of old capabilities.", RLVR_COLOR),
            ("Diverse alignment",
             "Merge models fine-tuned on different cultural\nor value preferences (Sakana Labs).", MERGE_COLOR),
        ]

        use_groups = VGroup()
        for name, desc, color in uses:
            n = Text(name, font_size=21, weight=BOLD, color=color)
            d = Text(desc, font_size=18, color=GRAY_B, line_spacing=1.3)
            d.next_to(n, DOWN, buff=0.08, aligned_edge=LEFT)
            use_groups.add(VGroup(n, d))

        use_groups.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        use_groups.next_to(use_title, DOWN, buff=0.5, aligned_edge=LEFT).shift(LEFT * 1)

        self.play(Write(use_title))
        for ug in use_groups:
            self.play(FadeIn(ug, shift=UP * 0.1), run_time=0.7)
        self.next_slide()

        # ── Beat 8: The shift ────────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects])

        shift = Text(
            "Model merging is emerging as a core tool\n"
            "in model development.",
            font_size=32, weight=BOLD, line_spacing=1.4,
        )

        self.play(FadeIn(shift, scale=1.1), run_time=1.0)
        self.next_slide()

