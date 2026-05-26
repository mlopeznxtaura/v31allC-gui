"""gui/stage3/view.py — V31all Stage 3 GUI
Three equal-focus panels:

  PANEL A — Inference Engine
    Single-turn and multi-turn inference via V31InferenceEngine.
    Shows all 9 output fields: turn, V, NFP, LTO, gate_status,
    M1, M2, M3, binary/geometry/language scalars.
    V(s) trajectory chart, gate OPEN/BUILDING indicator.

  PANEL B — Corpus Replay
    Load an existing v31 JSONL corpus, replay all records through
    inference engine, display per-record results, export replay log.

  PANEL C — Telemetry
    Read and display phase_gate_v31.jsonl live.
    Gate open/building ratio. V series over gate checks.
    Condition hit rates (v_ok, action_ok, m2_ok, m3_ok).
    Export telemetry snapshot.
"""

from nicegui import ui
from stage3.inference_engine.main import V31InferenceEngine
from stage2.dag_compiler.phase_gate import CONVERGENCE_ACTIONS
from stage1.core.triangulation import ACTION_VOCAB
import json, time, math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR   = ROOT_DIR / "output"
TELEMETRY_DIR = ROOT_DIR / "telemetry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── module-level state ─────────────────────────────────────────────────────────
_engine: V31InferenceEngine | None = None
_infer_history: list[dict] = []
_replay_results: list[dict] = []


def _get_engine() -> V31InferenceEngine:
    global _engine
    if _engine is None:
        _engine = V31InferenceEngine()
    return _engine


def _reset_engine():
    global _engine
    _engine = None


def _gate_color(status: str) -> str:
    return "text-emerald-600" if status == "OPEN" else "text-amber-600"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 VIEW
# ─────────────────────────────────────────────────────────────────────────────
def stage3_view() -> None:
    with ui.column().classes("w-full gap-4"):

        ui.label("🔮 Stage 3 — Inference & Telemetry").classes("text-lg font-bold text-rose-700")
        ui.label(
            "Stateful inference engine, corpus replay, live telemetry."
        ).classes("text-xs text-slate-500")

        # ══════════════════════════════════════════════════════════════════════
        # PANEL A — Inference Engine
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-rose-300 p-3"):
            ui.label("🧠 Inference Engine").classes("font-semibold text-rose-700 text-sm")
            ui.label(
                "7-input stateful inference. M1/M2/M3 + Triangulation state persist across turns."
            ).classes("text-xs text-slate-500 mb-2")

            # inputs
            with ui.column().classes("w-full gap-2"):
                infer_binary   = ui.textarea(
                    label="Binary text",
                    value="Phase gate logical: hash=v31all, mission=NextAura ATP inference, score=0.891"
                ).classes("w-full h-16")
                infer_geometry = ui.textarea(
                    label="Geometry text",
                    value="Entry 1. Env=Sandbox. SystemID=CASEBELIZE. TelemetryStable=True."
                ).classes("w-full h-16")
                infer_language = ui.textarea(
                    label="Language text",
                    value="Running v31 inference. 7-input schema active. Triangulation receiving all scalars."
                ).classes("w-full h-16")

            with ui.row().classes("gap-3 items-end mt-1"):
                infer_stim   = ui.number(label="Stimulus", value=1.0, min=0.1, max=10.0, step=0.1).classes("w-24")
                infer_btn    = ui.button("▶ Infer").props("dense color=rose")
                reset_btn    = ui.button("↺ Reset Engine").props("dense outline color=rose")

            # output fields — 9 fields
            ui.label("Output:").classes("text-xs font-semibold text-slate-600 mt-1")
            with ui.row().classes("gap-2 flex-wrap"):
                out_fields: dict[str, ui.label] = {}
                for key in ["turn", "V", "gate_status", "M1", "M2", "M3",
                            "binary_scalar", "geometry_scalar", "language_scalar"]:
                    with ui.card().classes("p-2 min-w-20"):
                        ui.label(key).classes("text-xs text-slate-500")
                        lbl = ui.label("—").classes("text-base font-mono font-bold")
                        out_fields[key] = lbl

            # NFP / LTO full text
            with ui.row().classes("gap-2 w-full mt-1"):
                nfp_card = ui.card().classes("flex-1 p-2")
                with nfp_card:
                    ui.label("next_frame_prediction").classes("text-xs text-slate-500")
                    nfp_lbl = ui.label("—").classes("text-xs font-mono")
                lto_card = ui.card().classes("flex-1 p-2")
                with lto_card:
                    ui.label("language_token_output").classes("text-xs text-slate-500")
                    lto_lbl = ui.label("—").classes("text-base font-mono font-bold text-rose-700")

            # V trajectory chart
            v_chart = ui.echart({
                "animation": False,
                "grid": {"top": 6, "bottom": 24, "left": 36, "right": 8},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 9}},
                "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"fontSize": 9}},
                "series": [{
                    "type": "line", "data": [], "smooth": True, "symbol": "circle", "symbolSize": 4,
                    "lineStyle": {"color": "#f43f5e", "width": 2},
                    "areaStyle": {"color": "rgba(244,63,94,0.08)"},
                }],
            }).classes("w-full h-32 mt-1")

            infer_log = ui.log(max_lines=12).classes(
                "w-full text-xs font-mono h-24 bg-slate-900 text-rose-300 mt-1"
            )

            def _run_infer():
                eng = _get_engine()
                try:
                    result = eng.infer(
                        binary_text=infer_binary.value,
                        geometry_text=infer_geometry.value,
                        language_text=infer_language.value,
                        stimulus=float(infer_stim.value),
                    )
                    _infer_history.append(result)

                    for key, lbl in out_fields.items():
                        val = result.get(key, "—")
                        lbl.set_text(str(val))
                        if key == "gate_status":
                            lbl.classes(remove="text-emerald-600 text-amber-600")
                            lbl.classes(add=_gate_color(str(val)))

                    nfp_lbl.set_text(result.get("next_frame_prediction", "—"))
                    lto_lbl.set_text(result.get("language_token_output", "—"))

                    # update V chart
                    v_vals = [r["V"] for r in _infer_history[-30:]]
                    v_chart.options["xAxis"]["data"] = [str(i+1) for i in range(len(v_vals))]
                    v_chart.options["series"][0]["data"] = v_vals
                    v_chart.update()

                    gate = result["gate_status"]
                    infer_log.push(
                        f"[turn {result['turn']}] V={result['V']:.4f} "
                        f"gate={gate} lto={result['language_token_output']}"
                    )
                    infer_log.push(
                        f"  M1={result['M1']} M2={result['M2']} M3={result['M3']} "
                        f"bin={result['binary_scalar']} geo={result['geometry_scalar']} "
                        f"lang={result['language_scalar']}"
                    )
                except Exception as ex:
                    infer_log.push(f"✗ Inference error: {ex}")

            def _reset():
                _reset_engine()
                _infer_history.clear()
                for lbl in out_fields.values():
                    lbl.set_text("—")
                nfp_lbl.set_text("—")
                lto_lbl.set_text("—")
                v_chart.options["xAxis"]["data"] = []
                v_chart.options["series"][0]["data"] = []
                v_chart.update()
                infer_log.push("↺ Engine reset — new M1/M2/M3/Triangulation state.")

            def _export_infer():
                if not _infer_history:
                    infer_log.push("No inference history to export.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"inference_log_{ts}.jsonl"
                out.write_text("\n".join(json.dumps(r) for r in _infer_history))
                infer_log.push(f"✅ Exported {len(_infer_history)} turns → {out.name}")

            infer_btn.on("click", lambda: _run_infer())
            reset_btn.on("click", lambda: _reset())
            ui.button("Export Inference Log", on_click=_export_infer).props("dense outline color=rose")

        # ══════════════════════════════════════════════════════════════════════
        # PANEL B — Corpus Replay
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-indigo-300 p-3"):
            ui.label("▶▶ Corpus Replay").classes("font-semibold text-indigo-700 text-sm")
            ui.label(
                "Load a v31 JSONL corpus and replay every record through a fresh inference engine."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3 items-end"):
                replay_path_input = ui.input(
                    label="Corpus JSONL path",
                    value=str(OUTPUT_DIR / "corpus_generated.jsonl"),
                ).classes("flex-1")
                replay_btn = ui.button("▶ Replay Corpus").props("dense color=indigo")

            # replay stats
            with ui.row().classes("gap-4 mt-1"):
                replay_count_lbl = ui.label("records: —").classes("text-xs font-mono")
                replay_vmean_lbl = ui.label("V_mean: —").classes("text-xs font-mono")
                replay_gate_lbl  = ui.label("gate OPEN: —").classes("text-xs font-mono text-emerald-600")
                replay_lto_lbl   = ui.label("top action: —").classes("text-xs font-mono text-indigo-600")

            # replay V chart
            replay_v_chart = ui.echart({
                "animation": False,
                "grid": {"top": 6, "bottom": 24, "left": 36, "right": 8},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 9}},
                "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"fontSize": 9}},
                "series": [
                    {
                        "type": "line", "data": [], "smooth": True, "symbol": "none",
                        "lineStyle": {"color": "#6366f1", "width": 2},
                        "areaStyle": {"color": "rgba(99,102,241,0.08)"},
                        "name": "V",
                    }
                ],
            }).classes("w-full h-32 mt-1")

            replay_log = ui.log(max_lines=12).classes(
                "w-full text-xs font-mono h-24 bg-slate-900 text-indigo-300 mt-1"
            )

            def _replay_corpus():
                global _replay_results
                path_str = replay_path_input.value.strip()
                path = Path(path_str)
                if not path.exists():
                    # try output dir
                    alt = OUTPUT_DIR / path.name
                    if alt.exists():
                        path = alt
                    else:
                        replay_log.push(f"✗ File not found: {path_str}")
                        return

                replay_log.push(f"▶ Replaying {path.name}...")
                try:
                    eng = V31InferenceEngine()  # fresh engine per replay
                    results = eng.run_corpus(str(path))
                    _replay_results = results

                    v_vals    = [r["V"] for r in results]
                    v_mean    = sum(v_vals) / len(v_vals) if v_vals else 0.0
                    open_ct   = sum(1 for r in results if r["gate_status"] == "OPEN")
                    actions   = [r["language_token_output"] for r in results]
                    act_count: dict = {}
                    for a in actions:
                        act_count[a] = act_count.get(a, 0) + 1
                    top_action = max(act_count, key=act_count.get) if act_count else "—"

                    replay_count_lbl.set_text(f"records: {len(results)}")
                    replay_vmean_lbl.set_text(f"V_mean: {v_mean:.4f}")
                    replay_gate_lbl.set_text(f"gate OPEN: {open_ct}/{len(results)}")
                    replay_lto_lbl.set_text(f"top action: {top_action}")

                    replay_v_chart.options["xAxis"]["data"] = [str(i+1) for i in range(len(v_vals))]
                    replay_v_chart.options["series"][0]["data"] = [round(v, 4) for v in v_vals]
                    replay_v_chart.update()

                    replay_log.push(
                        f"✅ {len(results)} records | V_mean={v_mean:.4f} | "
                        f"gate_OPEN={open_ct} | top_action={top_action}"
                    )
                    for r in results[:5]:
                        replay_log.push(
                            f"  [turn {r['turn']}] V={r['V']:.4f} gate={r['gate_status']} "
                            f"lto={r['language_token_output']}"
                        )
                    if len(results) > 5:
                        replay_log.push(f"  ... {len(results)-5} more")
                except Exception as ex:
                    replay_log.push(f"✗ Replay error: {ex}")

            def _export_replay():
                if not _replay_results:
                    replay_log.push("No replay results to export.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"replay_results_{ts}.jsonl"
                # strip source_record to keep it compact
                slim = [{k: v for k, v in r.items() if k != "source_record"} for r in _replay_results]
                out.write_text("\n".join(json.dumps(r) for r in slim))
                replay_log.push(f"✅ Exported {len(slim)} replay records → {out.name}")

            replay_btn.on("click", lambda: _replay_corpus())
            ui.button("Export Replay Results", on_click=_export_replay).props(
                "dense outline color=indigo"
            )

        # ══════════════════════════════════════════════════════════════════════
        # PANEL C — Telemetry
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-slate-300 p-3"):
            ui.label("📡 Telemetry — Phase Gate Log").classes("font-semibold text-slate-700 text-sm")
            ui.label(
                "Live read of telemetry/phase_gate_v31.jsonl. Condition hit rates, V series, gate ratio."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3"):
                telem_load_btn   = ui.button("↻ Refresh Telemetry").props("dense color=slate")
                telem_export_btn = ui.button("Export Snapshot").props("dense outline color=slate")

            # summary stats
            with ui.row().classes("gap-4 mt-1"):
                t_total_lbl    = ui.label("checks: —").classes("text-xs font-mono")
                t_open_lbl     = ui.label("OPEN: —").classes("text-xs font-mono text-emerald-600")
                t_building_lbl = ui.label("BUILDING: —").classes("text-xs font-mono text-amber-600")
                t_ratio_lbl    = ui.label("open ratio: —").classes("text-xs font-mono")

            # condition hit rates
            ui.label("Condition hit rates:").classes("text-xs text-slate-500 mt-1")
            cond_bars: dict[str, tuple] = {}
            for cond in ["v_ok", "action_ok", "m2_ok", "m3_ok"]:
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(cond).classes("w-20 text-xs font-mono")
                    bar_wrap = ui.element("div").classes("flex-1 bg-slate-100 rounded h-3 overflow-hidden")
                    with bar_wrap:
                        bar = ui.element("div").classes("bg-emerald-500 h-full rounded").style("width: 0%")
                    rate_lbl = ui.label("—").classes("w-12 text-xs font-mono text-right")
                    cond_bars[cond] = (bar, rate_lbl)

            # V chart over gate checks
            telem_v_chart = ui.echart({
                "animation": False,
                "grid": {"top": 6, "bottom": 24, "left": 36, "right": 8},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 9}},
                "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"fontSize": 9}},
                "series": [
                    {
                        "type": "line", "data": [], "smooth": False, "symbol": "circle", "symbolSize": 3,
                        "lineStyle": {"color": "#64748b", "width": 1.5},
                        "name": "V",
                    },
                    {
                        "type": "line", "data": [], "smooth": False, "symbol": "none",
                        "lineStyle": {"color": "#10b981", "width": 1, "type": "dashed"},
                        "name": "threshold (0.75)",
                    },
                ],
            }).classes("w-full h-32 mt-1")

            telem_log = ui.log(max_lines=14).classes(
                "w-full text-xs font-mono h-28 bg-slate-900 text-slate-300 mt-1"
            )

            def _load_telemetry():
                path = TELEMETRY_DIR / "phase_gate_v31.jsonl"
                if not path.exists():
                    telem_log.push("No phase_gate_v31.jsonl found yet.")
                    return
                entries = []
                for line in path.read_text().strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass

                if not entries:
                    telem_log.push("Gate log is empty.")
                    return

                total  = len(entries)
                open_c = sum(1 for e in entries if e.get("status") == "OPEN")
                build_c = total - open_c
                ratio  = open_c / total if total else 0.0

                t_total_lbl.set_text(f"checks: {total}")
                t_open_lbl.set_text(f"OPEN: {open_c}")
                t_building_lbl.set_text(f"BUILDING: {build_c}")
                t_ratio_lbl.set_text(f"open ratio: {ratio:.2%}")

                # condition hit rates
                for cond in ["v_ok", "action_ok", "m2_ok", "m3_ok"]:
                    hit = sum(1 for e in entries if e.get(cond) is True)
                    rate = hit / total
                    bar, lbl = cond_bars[cond]
                    bar.style(f"width: {rate*100:.1f}%")
                    lbl.set_text(f"{rate:.0%}")

                # V series chart — last 50
                recent = entries[-50:]
                v_vals = [e.get("V", 0.0) for e in recent]
                x_data = [str(i+1) for i in range(len(v_vals))]
                telem_v_chart.options["xAxis"]["data"] = x_data
                telem_v_chart.options["series"][0]["data"] = [round(v, 4) for v in v_vals]
                telem_v_chart.options["series"][1]["data"] = [0.75] * len(v_vals)
                telem_v_chart.update()

                telem_log.push(
                    f"↻ Loaded {total} entries | OPEN={open_c} ({ratio:.0%}) | "
                    f"BUILDING={build_c}"
                )
                for e in entries[-6:]:
                    telem_log.push(
                        f"  [{e.get('status','?')}] V={e.get('V',0):.3f} "
                        f"M2={e.get('M2',0):.3f} M3={e.get('M3',0):.3f} "
                        f"act={e.get('action','?')}"
                    )

            def _export_telem():
                path = TELEMETRY_DIR / "phase_gate_v31.jsonl"
                if not path.exists():
                    telem_log.push("No telemetry to export.")
                    return
                ts  = int(time.time())
                out = OUTPUT_DIR / f"telemetry_snapshot_{ts}.jsonl"
                out.write_text(path.read_text())
                telem_log.push(f"✅ Snapshot → {out.name}")

            telem_load_btn.on("click", lambda: _load_telemetry())
            telem_export_btn.on("click", lambda: _export_telem())

            # auto-load on init
            _load_telemetry()
