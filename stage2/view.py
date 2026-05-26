"""gui/stage2/view.py — V31all Stage 2 GUI
Learned Triangulation Weights Visualization

Shows:
  - 7-weight vector editor (sliders, live renormalization)
  - V(s) trajectory chart (last 50 ticks)
  - Per-weight contribution bars (how much each weight drove the last V)
  - Weight deviation from DEFAULT_WEIGHTS
  - Import / Export weight configs as JSON
  - Live simulation: fire N ticks with current weights and show resulting V series
"""

from nicegui import ui
from stage1.core.triangulation import TriangulationState, DEFAULT_WEIGHTS
from stage1.core.m_scalars import M1State, M2State, M3State, compute_m_scalars
import json, time, math
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
WEIGHT_LABELS = ["M1", "M2", "M3", "Binary", "Geometry", "Language", "Momentum"]
WEIGHT_COLORS = [
    "text-cyan-600",
    "text-emerald-600",
    "text-rose-600",
    "text-violet-600",
    "text-amber-600",
    "text-sky-600",
    "text-slate-600",
]
BAR_COLORS = [
    "bg-cyan-500",
    "bg-emerald-500",
    "bg-rose-500",
    "bg-violet-500",
    "bg-amber-500",
    "bg-sky-500",
    "bg-slate-500",
]

OUTPUT_DIR = Path("/mnt/d/NextAura/v31all_1/v31allC/output")

# ── Module-level state ─────────────────────────────────────────────────────────
_weights = list(DEFAULT_WEIGHTS)           # mutable working copy
_v_history: list[float] = []              # V(s) trajectory
_contrib_history: list[list[float]] = []  # per-weight raw contributions (last tick)
_sim_log: list[str] = []


def _renormalize(weights: list[float]) -> list[float]:
    """Force weights to sum to 1.0."""
    s = sum(weights)
    if s < 1e-9:
        return [1.0 / len(weights)] * len(weights)
    return [w / s for w in weights]


def _simulate_ticks(n: int, stimulus: float = 1.0) -> tuple[list[float], list[list[float]]]:
    """Run n ticks with current _weights. Returns (v_series, contrib_series)."""
    m1 = M1State()
    m2 = M2State()
    m3 = M3State()
    tri = TriangulationState(weights=list(_weights))

    v_series = []
    contrib_series = []

    for _ in range(n):
        vals = compute_m_scalars(m1, m2, m3, stimulus=stimulus)
        M1v, M2v, M3v = vals["M1"], vals["M2"], vals["M3"]
        # geometry / language derived from scalar outputs (normalized)
        geom = min(max(M2v, 0.0), 1.0)
        lang = min(max(M3v / 2.0, 0.0), 1.0)
        binary_v = min(max(m1.branch_count / 100.0, 0.0), 1.0)

        v = tri.compute(M1v, M2v, M3v, binary_v, geom, lang)
        v_series.append(v)

        # per-weight raw contributions at this tick
        m1_norm = (math.tanh(M1v / max(abs(M1v) + 1e-8, 1.0)) + 1.0) / 2.0
        m3_norm = (min(max(M3v, 0.5), 2.0) - 0.5) / 1.5
        raw_inputs = [m1_norm, M2v, m3_norm, binary_v, geom, lang, tri._prior_v]
        contribs = [w * x for w, x in zip(_weights, raw_inputs)]
        contrib_series.append(contribs)

    return v_series, contrib_series


# ── Main Stage 2 View ──────────────────────────────────────────────────────────
def stage2_view() -> None:
    global _weights

    with ui.column().classes("w-full gap-4"):

        # ── Header ─────────────────────────────────────────────────────────────
        ui.label("⚖ Triangulation Weight Editor").classes("text-lg font-bold text-emerald-700")
        ui.label(
            "Tune the 7 triangulation weights. Sum auto-renormalizes to 1.0. "
            "Run simulation to see V(s) trajectory and per-weight impact."
        ).classes("text-xs text-slate-500")

        # ── Weight Sliders ─────────────────────────────────────────────────────
        with ui.card().classes("w-full p-4 border border-emerald-300"):
            ui.label("Weight Vector").classes("font-semibold text-emerald-700 text-sm mb-2")
            sliders = []
            slider_labels = []
            default_labels = []

            for i, (label, color, default) in enumerate(zip(WEIGHT_LABELS, WEIGHT_COLORS, DEFAULT_WEIGHTS)):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(label).classes(f"w-20 text-xs font-mono font-bold {color}")
                    sl = ui.slider(min=0.0, max=1.0, step=0.01, value=_weights[i]).classes("flex-1")
                    val_lbl = ui.label(f"{_weights[i]:.3f}").classes("w-12 text-xs font-mono text-right")
                    dev_lbl = ui.label("±0.000").classes("w-16 text-xs font-mono text-slate-400")
                    sliders.append(sl)
                    slider_labels.append(val_lbl)
                    default_labels.append((dev_lbl, default))

            def _on_slider_change():
                raw = [sl.value for sl in sliders]
                norm = _renormalize(raw)
                _weights[:] = norm
                for i2, (sl2, lbl, (dev_lbl, default)) in enumerate(
                    zip(sliders, slider_labels, default_labels)
                ):
                    lbl.set_text(f"{norm[i2]:.3f}")
                    dev = norm[i2] - default
                    dev_lbl.set_text(f"{dev:+.3f}")

            for sl in sliders:
                sl.on("update:model-value", lambda _: _on_slider_change())

        # ── Preset Buttons ─────────────────────────────────────────────────────
        with ui.row().classes("gap-2"):
            def _apply_preset(preset: list[float]):
                norm = _renormalize(preset)
                _weights[:] = norm
                for i2, sl in enumerate(sliders):
                    sl.value = norm[i2]
                _on_slider_change()

            ui.button("Default", on_click=lambda: _apply_preset(list(DEFAULT_WEIGHTS))).props("dense color=emerald outline")
            ui.button("M1-Heavy", on_click=lambda: _apply_preset([0.40, 0.10, 0.15, 0.10, 0.10, 0.08, 0.07])).props("dense color=cyan outline")
            ui.button("M2-Efficient", on_click=lambda: _apply_preset([0.10, 0.40, 0.15, 0.10, 0.10, 0.08, 0.07])).props("dense color=emerald outline")
            ui.button("Balanced-7", on_click=lambda: _apply_preset([1/7]*7)).props("dense color=slate outline")
            ui.button("Self-Reinforce", on_click=lambda: _apply_preset([0.15, 0.12, 0.15, 0.10, 0.10, 0.10, 0.28])).props("dense color=violet outline")

        ui.separator()

        # ── Simulation Controls ────────────────────────────────────────────────
        ui.label("Simulation").classes("font-semibold text-emerald-700 text-sm")
        with ui.row().classes("gap-3 items-center"):
            n_ticks_input = ui.number(label="Ticks", value=20, min=1, max=200, step=1).classes("w-24")
            stim_input = ui.number(label="Stimulus", value=1.0, min=0.1, max=10.0, step=0.1).classes("w-28")
            run_btn = ui.button("▶ Run Simulation").props("color=emerald")

        # ── V(s) Trajectory Chart ──────────────────────────────────────────────
        with ui.card().classes("w-full p-3 border border-emerald-200"):
            ui.label("V(s) Trajectory").classes("text-xs font-semibold text-slate-600 mb-2")
            v_chart = ui.echart({
                "animation": False,
                "grid": {"top": 10, "bottom": 24, "left": 36, "right": 8},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 9}},
                "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"fontSize": 9}},
                "series": [{
                    "type": "line",
                    "data": [],
                    "smooth": True,
                    "lineStyle": {"color": "#10b981", "width": 2},
                    "areaStyle": {"color": "rgba(16,185,129,0.1)"},
                    "symbol": "none",
                }],
            }).classes("w-full h-36")

        # ── Per-Weight Contribution Bars ───────────────────────────────────────
        with ui.card().classes("w-full p-3 border border-emerald-200"):
            ui.label("Last-Tick Weight Contributions").classes("text-xs font-semibold text-slate-600 mb-2")
            contrib_bars = []
            contrib_val_labels = []
            for i, (label, bar_color) in enumerate(zip(WEIGHT_LABELS, BAR_COLORS)):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(label).classes("w-16 text-xs font-mono")
                    bar_wrap = ui.element("div").classes("flex-1 bg-slate-100 rounded h-4 overflow-hidden")
                    with bar_wrap:
                        bar = ui.element("div").classes(f"{bar_color} h-full rounded").style("width: 0%")
                    val_lbl = ui.label("0.000").classes("w-12 text-xs font-mono text-right")
                    contrib_bars.append(bar)
                    contrib_val_labels.append(val_lbl)

        # ── Summary Row ────────────────────────────────────────────────────────
        with ui.row().classes("gap-4 w-full"):
            with ui.card().classes("flex-1 p-3"):
                ui.label("Final V(s)").classes("text-xs text-slate-500")
                final_v_label = ui.label("—").classes("text-2xl font-mono font-bold text-emerald-700")
            with ui.card().classes("flex-1 p-3"):
                ui.label("Mean V").classes("text-xs text-slate-500")
                mean_v_label = ui.label("—").classes("text-2xl font-mono font-bold")
            with ui.card().classes("flex-1 p-3"):
                ui.label("V Std Dev").classes("text-xs text-slate-500")
                std_v_label = ui.label("—").classes("text-2xl font-mono font-bold")
            with ui.card().classes("flex-1 p-3"):
                ui.label("Dominant Weight").classes("text-xs text-slate-500")
                dom_weight_label = ui.label("—").classes("text-lg font-mono font-bold text-emerald-600")

        # ── Activity Log ───────────────────────────────────────────────────────
        sim_log = ui.log(max_lines=20).classes("w-full text-xs font-mono h-28 bg-slate-900 text-emerald-300")

        # ── Import / Export ────────────────────────────────────────────────────
        ui.separator()
        ui.label("Weight Config I/O").classes("font-semibold text-sm text-slate-600")
        with ui.row().classes("gap-2"):

            def _export_weights():
                ts = int(time.time())
                cfg = {
                    "timestamp": ts,
                    "weights": dict(zip(WEIGHT_LABELS, _weights)),
                    "raw_list": list(_weights),
                    "sum": round(sum(_weights), 6),
                }
                out = OUTPUT_DIR / f"weights_config_{ts}.json"
                out.write_text(json.dumps(cfg, indent=2))
                sim_log.push(f"✅ Weights exported → {out.name}")

            def _import_weights(e):
                try:
                    content = e.content.read().decode()
                    cfg = json.loads(content)
                    raw = cfg.get("raw_list") or list(cfg.get("weights", {}).values())
                    if len(raw) != 7:
                        sim_log.push("✗ Import error: need exactly 7 weights")
                        return
                    _apply_preset(raw)
                    sim_log.push(f"✅ Weights imported from {e.name}")
                except Exception as ex:
                    sim_log.push(f"✗ Import error: {ex}")

            ui.button("Export Weights JSON", on_click=_export_weights).props("dense color=emerald outline")
            ui.upload(on_upload=_import_weights).props("accept=.json dense label='Import JSON'").classes("text-xs")

        # ── Run Button Logic ───────────────────────────────────────────────────
        def _run_simulation():
            global _v_history, _contrib_history
            n = int(n_ticks_input.value or 20)
            stim = float(stim_input.value or 1.0)

            sim_log.push(f"▶ Running {n} ticks | stimulus={stim:.2f} | weights={[round(w,3) for w in _weights]}")

            v_series, contrib_series = _simulate_ticks(n, stimulus=stim)
            _v_history = v_series[-50:]  # keep last 50
            _contrib_history = contrib_series

            # Update V(s) chart
            x_data = [str(i + 1) for i in range(len(_v_history))]
            v_chart.options["xAxis"]["data"] = x_data
            v_chart.options["series"][0]["data"] = [round(v, 4) for v in _v_history]
            v_chart.update()

            # Update contribution bars from last tick
            last_contribs = contrib_series[-1] if contrib_series else [0.0] * 7
            max_c = max(last_contribs) if max(last_contribs) > 1e-9 else 1.0
            for i2, (bar, val_lbl) in enumerate(zip(contrib_bars, contrib_val_labels)):
                pct = (last_contribs[i2] / max_c) * 100.0
                bar.style(f"width: {pct:.1f}%")
                val_lbl.set_text(f"{last_contribs[i2]:.4f}")

            # Summary stats
            if v_series:
                final_v = v_series[-1]
                mean_v = sum(v_series) / len(v_series)
                variance = sum((v - mean_v) ** 2 for v in v_series) / len(v_series)
                std_v = math.sqrt(variance)
                dom_idx = last_contribs.index(max(last_contribs))
                dom_name = WEIGHT_LABELS[dom_idx]

                final_v_label.set_text(f"{final_v:.4f}")
                mean_v_label.set_text(f"{mean_v:.4f}")
                std_v_label.set_text(f"{std_v:.4f}")
                dom_weight_label.set_text(dom_name)

                sim_log.push(
                    f"  V_final={final_v:.4f}  mean={mean_v:.4f}  std={std_v:.4f}  dom={dom_name}"
                )

        run_btn.on("click", lambda: _run_simulation())
