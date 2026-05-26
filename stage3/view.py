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
from stage3.inference_engine.neural_inference import V31NeuralInferenceEngine
from stage2.dag_compiler.phase_gate import CONVERGENCE_ACTIONS
from stage1.core.triangulation import ACTION_VOCAB
import json, time, math, asyncio
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR   = ROOT_DIR / "output"
TELEMETRY_DIR = ROOT_DIR / "telemetry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── module-level state ─────────────────────────────────────────────────────────
_classical_engine: V31InferenceEngine | None = None
_neural_engine: V31NeuralInferenceEngine | None = None
_infer_history: list[dict] = []
_replay_results: list[dict] = []
_last_record_mode_index = 0


def _get_active_engine(is_neural: bool) -> V31InferenceEngine | V31NeuralInferenceEngine:
    global _classical_engine, _neural_engine
    if is_neural:
        if _neural_engine is None:
            _neural_engine = V31NeuralInferenceEngine(checkpoint_path="models/v31_neural_model.pt")
        # seamless hot-reload weights
        _neural_engine.load_weights("models/v31_neural_model.pt")
        return _neural_engine
    else:
        if _classical_engine is None:
            _classical_engine = V31InferenceEngine()
        return _classical_engine


def _reset_engines():
    global _classical_engine, _neural_engine
    _classical_engine = None
    _neural_engine = None


def generate_svg_grid(pixels: list[float], is_neural: bool = False) -> str:
    if not pixels or len(pixels) != 224:
        pixels = [0.0] * 224
        
    rows, cols = 14, 16
    rects = []
    for r in range(rows):
        for c in range(cols):
            val = pixels[r * cols + c]
            val_clamped = min(max(val, 0.0), 255.0)
            
            if is_neural:
                # Glowing violet-purple monochrome scale
                red = int(val_clamped * 0.65)
                green = int(val_clamped * 0.25)
                blue = int(val_clamped)
            else:
                # Glowing amber monochrome scale
                red = int(val_clamped)
                green = int(val_clamped * 0.70)
                blue = int(val_clamped * 0.15)
                
            rects.append(
                f'<rect x="{c}" y="{r}" width="0.9" height="0.9" rx="0.15" '
                f'fill="rgb({red},{green},{blue})" />'
            )
            
    svg_body = "\n".join(rects)
    return (
        f'<svg viewBox="0 0 16 14" width="100%" height="100%" '
        f'style="background-color: #030712; border-radius: 4px; padding: 2px;">\n'
        f'{svg_body}\n'
        f'</svg>'
    )


def generate_classical_pattern(v: float) -> list[float]:
    pattern = []
    for r in range(14):
        for c in range(16):
            dist = math.sqrt((r - 6.5)**2 + (c - 7.5)**2)
            val = 127.0 + 127.0 * math.sin(dist - v * 20.0)
            pattern.append(min(max(val, 0.0), 255.0))
    return pattern


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
            with ui.row().classes("gap-3 items-center w-full justify-between mb-1"):
                ui.label("🧠 Inference Engine").classes("font-semibold text-rose-700 text-sm")
                engine_mode_toggle = ui.toggle({False: "Analytical ⚪", True: "Neural (PyTorch) 🟣"}, value=False).props("unelevated toggle-color=rose text-xs")
            
            ui.label(
                "7-input stateful inference. M1/M2/M3 + Triangulation state persist across turns."
            ).classes("text-xs text-slate-500 mb-2")

            chat_messages: list[dict] = []
            chat_container_element = [None] # list container for closure mutable assignment

            # Tab layout for classic vs chat mode
            with ui.tabs().classes("w-full border-b border-rose-100 mb-2") as tabs:
                classic_tab = ui.tab("Structured Parameters ⚙")
                chat_tab = ui.tab("Interactive Chat Arena 💬")

            with ui.tab_panels(tabs, value=classic_tab).classes("w-full bg-transparent p-0"):
                with ui.tab_panel(classic_tab).classes("p-0 w-full gap-2"):
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
                        oneshot_infer_btn = ui.button("⚡ 1-SHOT AUTO-INFER").props("dense color=pink icon=bolt").classes("font-bold")
                        reset_btn    = ui.button("↺ Reset Engine").props("dense outline color=rose")

                with ui.tab_panel(chat_tab).classes("p-0 w-full gap-2"):
                    ui.label("Talk to the model directly. Your message overrides the Language Text parameter while inheriting current Binary and Geometry contexts.").classes("text-[11px] text-slate-500 mb-1")
                    
                    # Scrollable chat logs area
                    chat_scroll = ui.scroll_area().classes("w-full h-80 bg-slate-950 rounded-lg p-2 border border-slate-800")
                    with chat_scroll:
                        chat_container = ui.column().classes("w-full gap-2")
                        chat_container_element[0] = chat_container
                    
                    with ui.row().classes("w-full items-center gap-2 mt-2"):
                        chat_input = ui.input(placeholder="Type a message or question for the model (e.g. 'Are we stable?', 'What is our V(s) score?')...").classes("flex-1")
                        send_btn = ui.button(icon="send").props("color=rose flat")

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

            # NFP / LTO full text & Synthesized Visual Grid
            with ui.row().classes("gap-2 w-full mt-1 items-stretch"):
                nfp_card = ui.card().classes("flex-1 p-2")
                with nfp_card:
                    ui.label("next_frame_prediction").classes("text-xs text-slate-500")
                    nfp_lbl = ui.label("—").classes("text-xs font-mono")
                
                # Live 14x16 Matrix Canvas
                canvas_card = ui.card().classes("w-44 p-2 items-center justify-center bg-slate-950 border border-slate-800")
                with canvas_card:
                    ui.label("synthesized_visual_frame").classes("text-xs text-slate-400 mb-1 font-mono")
                    visual_grid_html = ui.html(generate_svg_grid([0.0]*224, is_neural=False)).classes("w-36 h-30")
                
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

            def _update_chat_ui():
                container = chat_container_element[0]
                if not container:
                    return
                container.clear()
                with container:
                    if not chat_messages:
                        ui.label("No conversation history yet. Ask the model anything!").classes(
                            "text-xs text-slate-500 italic text-center w-full my-4"
                        )
                        return
                        
                    for msg in chat_messages:
                        if msg["sender"] == "user":
                            # User bubble
                            with ui.row().classes("w-full justify-end gap-2 my-1"):
                                with ui.card().classes("bg-gradient-to-r from-rose-700 to-pink-600 text-white p-2.5 rounded-2xl rounded-tr-none max-w-[80%] shadow-md border-0"):
                                    ui.label(msg["text"]).classes("text-xs font-semibold leading-relaxed break-words")
                                    ui.label("User").classes("text-[9px] text-pink-200 mt-0.5 text-right block")
                        else:
                            # Model bubble
                            with ui.row().classes("w-full justify-start gap-2 my-1"):
                                bg_class = "bg-slate-900 border border-rose-900 text-rose-100" if "is_error" not in msg else "bg-red-950 border border-red-800 text-red-200"
                                with ui.card().classes(f"{bg_class} p-2.5 rounded-2xl rounded-tl-none max-w-[85%] shadow-md"):
                                    if "mode" in msg:
                                        with ui.row().classes("items-center justify-between w-full mb-1 border-b border-slate-800 pb-1 gap-4"):
                                            ui.label(msg["mode"]).classes("text-[9px] uppercase font-bold text-rose-400 tracking-wider")
                                            ui.label(f"Turn {msg['turn']}").classes("text-[9px] text-slate-500 font-mono")
                                    
                                    ui.label(msg["text"]).classes("text-xs font-semibold leading-relaxed break-words text-rose-50")
                                    
                                    if "v" in msg:
                                        with ui.expansion("Thought Trace & Frame").classes("w-full text-[11px] text-slate-400 mt-1.5 bg-slate-950 rounded p-1"):
                                            with ui.row().classes("w-full gap-2 justify-between mb-1 pb-1 border-b border-slate-900"):
                                                with ui.column().classes("gap-0.5"):
                                                    ui.label(f"V(s): {msg['v']:.4f}").classes("font-mono text-emerald-400 font-bold")
                                                    ui.label(f"Gate: {msg['gate_status']}").classes(f"font-mono font-bold {'text-emerald-500' if msg['gate_status'] == 'OPEN' else 'text-amber-500'}")
                                                with ui.column().classes("gap-0.5 items-end"):
                                                    ui.label(f"M1={msg['m1']:.2f} M2={msg['m2']:.2f}").classes("font-mono text-slate-500")
                                                    ui.label(f"M3={msg['m3']:.2f}").classes("font-mono text-slate-500")
                                            
                                            ui.label("Next Frame Prediction:").classes("text-[9px] text-slate-500 uppercase font-bold tracking-wider")
                                            ui.label(msg["nfp"]).classes("text-[10px] font-mono text-slate-300 bg-slate-900 p-1 rounded mb-1.5 leading-tight")
                                            
                                            ui.label("Synthesized Frame:").classes("text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1")
                                            # Inline mini canvas grid
                                            ui.html(generate_svg_grid(msg["pixels"], is_neural="Neural" in msg["mode"])).classes("w-24 h-20 mx-auto mb-1")
                                            
                                    ui.label("Model").classes("text-[9px] text-rose-500 mt-0.5 text-left block font-bold")
                
                # Auto-scroll container
                ui.run_javascript(f"var el = document.getElementById('{container.id}'); if(el) {{ el.scrollTop = el.scrollHeight; }}")

            async def _send_chat_message(user_msg_input):
                msg_text = user_msg_input.value.strip()
                if not msg_text:
                    return
                user_msg_input.set_value("")
                
                chat_messages.append({"sender": "user", "text": msg_text, "timestamp": time.time()})
                _update_chat_ui()
                
                is_n = bool(engine_mode_toggle.value)
                eng = _get_active_engine(is_n)
                try:
                    result = eng.infer(
                        binary_text=infer_binary.value,
                        geometry_text=infer_geometry.value,
                        language_text=msg_text,
                        stimulus=float(infer_stim.value),
                    )
                    _infer_history.append(result)
                    
                    # Update Panel A's numeric output labels
                    for key, lbl in out_fields.items():
                        val = result.get(key, "—")
                        lbl.set_text(str(val))
                        if key == "gate_status":
                            lbl.classes(remove="text-emerald-600 text-amber-600")
                            lbl.classes(add=_gate_color(str(val)))
                    
                    nfp_lbl.set_text(result.get("next_frame_prediction", "—"))
                    lto_lbl.set_text(result.get("language_token_output", "—"))
                    
                    # Update Panel A visual grid
                    if is_n and "neural_frame" in result:
                        grid_pixels = result["neural_frame"]
                    else:
                        grid_pixels = generate_classical_pattern(result["V"])
                    visual_grid_html.set_content(generate_svg_grid(grid_pixels, is_neural=is_n))
                    
                    # Update Panel A charts
                    v_vals = [r["V"] for r in _infer_history[-30:]]
                    v_chart.options["xAxis"]["data"] = [str(i+1) for i in range(len(v_vals))]
                    v_chart.options["series"][0]["data"] = v_vals
                    v_chart.update()
                    
                    # Log activity
                    gate = result["gate_status"]
                    mode_lbl = "NEURAL" if is_n else "CLASSICAL"
                    infer_log.push(f"[{mode_lbl} | turn {result['turn']}] V={result['V']:.4f} gate={gate} lto={result['language_token_output']}")
                    
                    # Add Model response bubble
                    chat_messages.append({
                        "sender": "model",
                        "mode": "Neural 🟣" if is_n else "Analytical ⚪",
                        "text": result.get("language_token_output", "—"),
                        "nfp": result.get("next_frame_prediction", "—"),
                        "v": result.get("V", 0.0),
                        "gate_status": gate,
                        "m1": result.get("M1", 0.0),
                        "m2": result.get("M2", 0.0),
                        "m3": result.get("M3", 0.0),
                        "binary_scalar": result.get("binary_scalar", 0.0),
                        "geometry_scalar": result.get("geometry_scalar", 0.0),
                        "language_scalar": result.get("language_scalar", 0.0),
                        "turn": result.get("turn", 0),
                        "pixels": grid_pixels,
                        "timestamp": time.time()
                    })
                    _update_chat_ui()
                except Exception as ex:
                    chat_messages.append({"sender": "model", "text": f"Error running inference: {ex}", "is_error": True})
                    _update_chat_ui()

            # Wire up chat triggers
            chat_input.on("keydown.enter", lambda: _send_chat_message(chat_input))
            send_btn.on("click", lambda: _send_chat_message(chat_input))
            _update_chat_ui()

            def _run_infer():
                is_n = bool(engine_mode_toggle.value)
                eng = _get_active_engine(is_n)
                try:
                    if is_n and not getattr(eng, "is_loaded", False):
                        ui.notify("Neural model weights not found. Running wave fallback.", type="warning")

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

                    # update visual grid
                    if is_n and "neural_frame" in result:
                        grid_pixels = result["neural_frame"]
                    else:
                        grid_pixels = generate_classical_pattern(result["V"])
                    visual_grid_html.set_content(generate_svg_grid(grid_pixels, is_neural=is_n))

                    # update V chart
                    v_vals = [r["V"] for r in _infer_history[-30:]]
                    v_chart.options["xAxis"]["data"] = [str(i+1) for i in range(len(v_vals))]
                    v_chart.options["series"][0]["data"] = v_vals
                    v_chart.update()

                    gate = result["gate_status"]
                    mode_lbl = "NEURAL" if is_n else "CLASSICAL"
                    infer_log.push(
                        f"[{mode_lbl} | turn {result['turn']}] V={result['V']:.4f} "
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
                _reset_engines()
                _infer_history.clear()
                chat_messages.clear()
                _update_chat_ui()
                for lbl in out_fields.values():
                    lbl.set_text("—")
                nfp_lbl.set_text("—")
                lto_lbl.set_text("—")
                visual_grid_html.set_content(generate_svg_grid([0.0]*224, is_neural=False))
                v_chart.options["xAxis"]["data"] = []
                v_chart.options["series"][0]["data"] = []
                v_chart.update()
                infer_log.push("↺ Engines reset — fresh classical & neural states.")

            def _export_infer():
                if not _infer_history:
                    infer_log.push("No inference history to export.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"inference_log_{ts}.jsonl"
                out.write_text("\n".join(json.dumps(r) for r in _infer_history))
                infer_log.push(f"[OK] Exported {len(_infer_history)} turns -> {out.name}")

            def _export_model_state():
                try:
                    from stage3.inference_engine.exporter import export_model
                    is_n = bool(engine_mode_toggle.value)
                    eng = _get_active_engine(is_n)
                    # Exporter takes Analytical V31InferenceEngine (or subclass)
                    paths = export_model(eng, OUTPUT_DIR)
                    infer_log.push(f"[OK] Exported GGUF Model Weights: {paths['gguf'].name} ({paths['gguf'].stat().st_size} bytes)")
                    infer_log.push(f"[OK] Exported JSON Model Parameters: {paths['json'].name} ({paths['json'].stat().st_size} bytes)")
                except Exception as ex:
                    infer_log.push(f"✗ Export GGUF Error: {ex}")

            def _export_pytorch_model():
                try:
                    import shutil
                    src = ROOT_DIR / "models" / "v31_neural_model.pt"
                    if not src.exists():
                        infer_log.push("✗ Export PyTorch Error: PyTorch model file 'models/v31_neural_model.pt' not found. Please train the model in Stage 2 first.")
                        return
                    ts = int(time.time())
                    dst = OUTPUT_DIR / f"model_export_v31_{ts}.pt"
                    shutil.copy(src, dst)
                    infer_log.push(f"[OK] Exported PyTorch Model Weights: {dst.name} ({dst.stat().st_size} bytes)")
                except Exception as ex:
                    infer_log.push(f"✗ Export PyTorch Error: {ex}")

            def _one_shot_infer():
                engine_mode_toggle.set_value(True)
                _get_active_engine(is_neural=True)
                _run_infer()

            oneshot_infer_btn.on("click", _one_shot_infer)
            infer_btn.on("click", lambda: _run_infer())
            reset_btn.on("click", lambda: _reset())
            
            with ui.row().classes("gap-2 mt-1"):
                ui.button("Export Inference Log", on_click=_export_infer).props("dense outline color=rose")
                ui.button("Export Model (.gguf)", on_click=_export_model_state).props("dense outline color=rose")
                ui.button("Export Model (.pt)", on_click=_export_pytorch_model).props("dense outline color=rose")

        # ══════════════════════════════════════════════════════════════════════
        # PANEL B — Corpus Replay
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-indigo-300 p-3"):
            with ui.row().classes("gap-3 items-center w-full justify-between mb-1"):
                ui.label("▶▶ Corpus Replay").classes("font-semibold text-indigo-700 text-sm")
                replay_mode_toggle = ui.toggle({False: "Analytical ⚪", True: "Neural (PyTorch) 🟣"}, value=False).props("unelevated toggle-color=indigo text-xs")
            
            ui.label(
                "Load a v31 JSONL corpus and replay every record through a fresh inference engine."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3 items-end w-full"):
                replay_path_input = ui.input(
                    label="Corpus JSONL path",
                    value="corpus_v31_sample.jsonl",
                ).classes("flex-1")
                replay_btn = ui.button("▶ Replay (Instant)").props("dense color=indigo")
                replay_live_btn = ui.button("▶ Replay (Live)").props("dense color=emerald")

            # Replay LED Dot Matrix Display!
            with ui.row().classes("w-full mt-1 items-center justify-center"):
                replay_canvas_card = ui.card().classes("w-44 p-2 items-center justify-center bg-slate-950 border border-slate-800")
                with replay_canvas_card:
                    ui.label("replay_visual_frame").classes("text-xs text-slate-400 mb-1 font-mono")
                    replay_grid_html = ui.html(generate_svg_grid([0.0]*224, is_neural=False)).classes("w-36 h-30")

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
                    # try root or output dir
                    alt = ROOT_DIR / path_str
                    if alt.exists():
                        path = alt
                    else:
                        alt_out = OUTPUT_DIR / Path(path_str).name
                        if alt_out.exists():
                            path = alt_out
                        else:
                            replay_log.push(f"✗ File not found: {path_str}")
                            return

                is_n = bool(replay_mode_toggle.value)
                mode_lbl = "Neural (PyTorch) 🟣" if is_n else "Classical ⚪"
                replay_log.push(f"▶ Replaying {path.name} using {mode_lbl}...")
                
                try:
                    records = []
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line: continue
                            try:
                                records.append(json.loads(line))
                            except Exception: continue
                    
                    if not records:
                        replay_log.push("✗ Corpus is empty or invalid.")
                        return

                    total = len(records)
                    if is_n:
                        eng = V31NeuralInferenceEngine(checkpoint_path="models/v31_neural_model.pt")
                    else:
                        eng = V31InferenceEngine()

                    results = []
                    v_vals = []
                    for idx, rec in enumerate(records):
                        si = rec.get("scalar_inputs", {})
                        binary_text   = si.get("binary", "")   or str(rec.get("M1", ""))
                        geometry_text = si.get("geometry", "") or str(rec.get("M2", ""))
                        language_text = si.get("language", "") or str(rec.get("M3", ""))

                        if len(geometry_text.strip()) < 20:
                            geometry_text = (
                                f"Entry {idx+1}/{total}. Env=Sandbox. SystemID=CASEBELIZE. "
                                f"TelemetryStable=True. {geometry_text}".strip()
                            )
                        math_rel = si.get("mathematical_relationship", "")
                        if len(language_text.strip()) < 20 and math_rel:
                            language_text = math_rel

                        res = eng.infer(
                            binary_text=binary_text,
                            geometry_text=geometry_text,
                            language_text=language_text,
                            stimulus=float(rec.get("stimulus", 1.0)),
                            is_logical=bool(rec.get("is_logical", True)),
                            telemetry_stable=bool(rec.get("telemetry_stable", True)),
                            entry_index=idx + 1,
                            total_entries=total,
                        )
                        res["source_record"] = rec
                        results.append(res)
                        v_vals.append(res["V"])

                    _replay_results = results
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

                    # Render last visual grid state
                    if is_n and "neural_frame" in results[-1]:
                        rep_pixels = results[-1]["neural_frame"]
                    else:
                        rep_pixels = generate_classical_pattern(results[-1]["V"])
                    replay_grid_html.set_content(generate_svg_grid(rep_pixels, is_neural=is_n))

                    replay_log.push(
                        f"[OK] Replayed {len(results)} records | V_mean={v_mean:.4f} | "
                        f"gate_OPEN={open_ct} | top_action={top_action}"
                    )
                except Exception as ex:
                    replay_log.push(f"✗ Replay error: {ex}")

            async def _replay_corpus_live():
                global _replay_results
                path_str = replay_path_input.value.strip()
                path = Path(path_str)
                if not path.exists():
                    # try root or output dir
                    alt = ROOT_DIR / path_str
                    if alt.exists():
                        path = alt
                    else:
                        alt_out = OUTPUT_DIR / Path(path_str).name
                        if alt_out.exists():
                            path = alt_out
                        else:
                            replay_log.push(f"✗ File not found: {path_str}")
                            return

                is_n = bool(replay_mode_toggle.value)
                mode_lbl = "Neural (PyTorch) 🟣" if is_n else "Classical ⚪"
                replay_log.push(f"▶ [LIVE] Replaying {path.name} using {mode_lbl}...")
                
                try:
                    records = []
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line: continue
                            try:
                                records.append(json.loads(line))
                            except Exception: continue

                    if not records:
                        replay_log.push("✗ Corpus is empty or invalid.")
                        return

                    total = len(records)
                    if is_n:
                        eng = V31NeuralInferenceEngine(checkpoint_path="models/v31_neural_model.pt")
                    else:
                        eng = V31InferenceEngine()

                    _replay_results = []
                    v_vals = []

                    for idx, rec in enumerate(records):
                        si = rec.get("scalar_inputs", {})
                        binary_text   = si.get("binary", "")   or str(rec.get("M1", ""))
                        geometry_text = si.get("geometry", "") or str(rec.get("M2", ""))
                        language_text = si.get("language", "") or str(rec.get("M3", ""))

                        if len(geometry_text.strip()) < 20:
                            geometry_text = (
                                f"Entry {idx+1}/{total}. Env=Sandbox. SystemID=CASEBELIZE. "
                                f"TelemetryStable=True. {geometry_text}".strip()
                            )
                        math_rel = si.get("mathematical_relationship", "")
                        if len(language_text.strip()) < 20 and math_rel:
                            language_text = math_rel

                        result = eng.infer(
                            binary_text=binary_text,
                            geometry_text=geometry_text,
                            language_text=language_text,
                            stimulus=float(rec.get("stimulus", 1.0)),
                            is_logical=bool(rec.get("is_logical", True)),
                            telemetry_stable=bool(rec.get("telemetry_stable", True)),
                            entry_index=idx + 1,
                            total_entries=total,
                        )
                        result["source_record"] = rec
                        _replay_results.append(result)

                        v_vals.append(result["V"])
                        v_mean = sum(v_vals) / len(v_vals)
                        open_ct = sum(1 for r in _replay_results if r["gate_status"] == "OPEN")

                        # Live update labels
                        replay_count_lbl.set_text(f"records: {idx+1}/{total}")
                        replay_vmean_lbl.set_text(f"V_mean: {v_mean:.4f}")
                        replay_gate_lbl.set_text(f"gate OPEN: {open_ct}/{idx+1}")
                        replay_lto_lbl.set_text(f"action: {result['language_token_output']}")

                        # Live update visual grid!
                        if is_n and "neural_frame" in result:
                            rep_pixels = result["neural_frame"]
                        else:
                            rep_pixels = generate_classical_pattern(result["V"])
                        replay_grid_html.set_content(generate_svg_grid(rep_pixels, is_neural=is_n))

                        # Live update chart
                        replay_v_chart.options["xAxis"]["data"] = [str(i+1) for i in range(len(v_vals))]
                        replay_v_chart.options["series"][0]["data"] = [round(v, 4) for v in v_vals]
                        replay_v_chart.update()

                        # Live push log
                        replay_log.push(
                            f"[{idx+1}/{total}] V={result['V']:.4f} gate={result['gate_status']} "
                            f"lto={result['language_token_output']}"
                        )

                        await asyncio.sleep(0.04)

                    replay_log.push(f"[OK] Live Replay Completed! {total} records successfully evaluated.")
                except Exception as ex:
                    replay_log.push(f"✗ Live Replay error: {ex}")

            def _export_replay():
                if not _replay_results:
                    replay_log.push("No replay results to export.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"replay_results_{ts}.jsonl"
                # strip source_record to keep it compact
                slim = [{k: v for k, v in r.items() if k != "source_record"} for r in _replay_results]
                out.write_text("\n".join(json.dumps(r) for r in slim))
                replay_log.push(f"[OK] Exported {len(slim)} replay records -> {out.name}")

            replay_btn.on("click", lambda: _replay_corpus())
            replay_live_btn.on("click", lambda: _replay_corpus_live())
            
            with ui.row().classes("gap-2 mt-1"):
                ui.button("Export Replay Results", on_click=_export_replay).props("dense outline color=indigo")

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

        # ══════════════════════════════════════════════════════════════════════
        # PANEL D — Screen Recorder
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-teal-300 p-3 mt-1"):
            with ui.row().classes("gap-3 items-center w-full justify-between mb-1"):
                ui.label("🎥 Live Screen Recorder").classes("font-semibold text-teal-700 text-sm")
            
            ui.label(
                "Record the host Windows desktop asynchronously using native FFmpeg."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3 items-end w-full flex-wrap"):
                rec_duration = ui.number(label="Duration (sec)", value=3.0, min=1.0, max=60.0, step=1.0).classes("w-24")
                rec_fps      = ui.number(label="Framerate (fps)", value=10.0, min=5.0, max=30.0, step=5.0).classes("w-24")
                rec_mode     = ui.select(
                    label="Recording Scope / Complexity", 
                    options=[
                        "Full Desktop 🖥", 
                        "Chrome Browser Only (1/3 Complexity) 🌐",
                        "Cycle Both Modes 🔀 (1,2,1,2...)"
                    ], 
                    value="Full Desktop 🖥"
                ).classes("w-72")
                rec_btn      = ui.button("▶ Start Recording").props("dense color=teal")

            # status & video preview
            rec_status = ui.label("Ready to capture").classes("text-xs font-mono text-teal-600 mt-1")
            
            video_container = ui.column().classes("w-full mt-2 hidden")
            with video_container:
                ui.label("Latest Recording Preview:").classes("text-xs text-slate-500 font-mono")
                video_player = ui.video("").classes("w-full h-48 bg-black rounded")
                download_link = ui.link("📥 Download recorded video (.mp4)", "#").classes("text-xs text-teal-600 font-bold")

            async def _start_recording():
                duration = float(rec_duration.value)
                fps = int(rec_fps.value)
                ts = int(time.time())
                output_filename = f"screen_rec_{ts}.mp4"
                
                # In WSL, the static path of the python app is:
                static_path = Path(__file__).resolve().parent.parent / "static"
                recordings_path = static_path / "recordings"
                recordings_path.mkdir(parents=True, exist_ok=True)
                
                output_file_wsl = recordings_path / output_filename
                
                # Convert the path to Windows format for ffmpeg.exe:
                abs_wsl_path = str(output_file_wsl.resolve())
                if abs_wsl_path.startswith("/mnt/"):
                    drive = abs_wsl_path[5].upper() # e.g. 'd' -> 'D'
                    win_path = f"{drive}:" + abs_wsl_path[6:]
                else:
                    win_path = abs_wsl_path.replace("\\", "/")
                
                rec_btn.disable()
                rec_status.set_text("🔴 INITIALIZING RECORDER...")
                rec_status.classes(remove="text-teal-600 text-rose-600")
                rec_status.classes(add="text-rose-600 font-bold")
                
                mode = rec_mode.value
                global _last_record_mode_index
                
                target_input = "desktop"
                actual_mode_name = "Full Desktop"
                
                if mode == "Chrome Browser Only (1/3 Complexity) 🌐":
                    target_input = "title=NextAura v31 TriSplit GUI"
                    actual_mode_name = "Chrome Window"
                elif mode == "Cycle Both Modes 🔀 (1,2,1,2...)":
                    if _last_record_mode_index % 2 == 0:
                        target_input = "desktop"
                        actual_mode_name = "Cycle (Full Desktop)"
                    else:
                        target_input = "title=NextAura v31 TriSplit GUI"
                        actual_mode_name = "Cycle (Chrome Window)"
                    _last_record_mode_index += 1
                
                cmd = [
                    "ffmpeg.exe",
                    "-f", "gdigrab",
                    "-framerate", str(fps),
                    "-i", target_input,
                    "-t", str(duration),
                    "-y", win_path
                ]
                
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # Live countdown in GUI
                    for sec in range(int(duration), 0, -1):
                        rec_status.set_text(f"🔴 RECORDING [{actual_mode_name}]... {sec}s remaining")
                        await asyncio.sleep(1.0)
                        
                    stdout, stderr = await proc.communicate()
                    returncode = proc.returncode
                    
                    # Fallback if window capture fails
                    if returncode != 0 and target_input != "desktop":
                        rec_status.set_text("⚠️ WINDOW CAPTURE FAILED. FALLING BACK TO DESKTOP...")
                        await asyncio.sleep(0.5)
                        
                        cmd_fallback = [
                            "ffmpeg.exe",
                            "-f", "gdigrab",
                            "-framerate", str(fps),
                            "-i", "desktop",
                            "-t", str(duration),
                            "-y", win_path
                        ]
                        proc_fb = await asyncio.create_subprocess_exec(
                            *cmd_fallback,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        for sec in range(int(duration), 0, -1):
                            rec_status.set_text(f"🔴 FALLBACK RECORDING DESKTOP... {sec}s remaining")
                            await asyncio.sleep(1.0)
                        stdout, stderr = await proc_fb.communicate()
                        returncode = proc_fb.returncode
                    
                    if returncode == 0 and output_file_wsl.exists():
                        rec_status.set_text(f"✓ Video recorded successfully! [{actual_mode_name}] ({output_filename})")
                        rec_status.classes(remove="text-rose-600")
                        rec_status.classes(add="text-emerald-600 font-bold")
                        
                        video_container.classes(remove="hidden")
                        video_url = f"/static/recordings/{output_filename}"
                        video_player.set_source(video_url)
                        download_link.props(f'href="{video_url}" target="_blank"')
                        ui.notify(f"Screen recording completed! [{actual_mode_name}]", type="positive")
                    else:
                        err_msg = stderr.decode(errors="ignore").strip().split("\n")[-1]
                        rec_status.set_text(f"✗ Recording failed: {err_msg}")
                        rec_status.classes(remove="text-rose-600 text-emerald-600")
                        rec_status.classes(add="text-rose-600 font-bold")
                        ui.notify(f"Screen recording failed: {err_msg}", type="negative")
                        
                except Exception as ex:
                    rec_status.set_text(f"✗ Recording exception: {ex}")
                    rec_status.classes(remove="text-rose-600 text-emerald-600")
                    rec_status.classes(add="text-rose-600 font-bold")
                    ui.notify(f"Screen recording error: {ex}", type="negative")
                finally:
                    rec_btn.enable()

            rec_btn.on("click", lambda: _start_recording())
