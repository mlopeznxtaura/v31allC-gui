"""gui/stage3/view.py — V31all Stage 3 GUI (Academic Research Theme)
Cohesive, state-of-the-art researcher dashboard focusing on slate/gray styling.

Three equal-focus research panels:

  PANEL A — Unified Inference Arena
    Integrated context inputs configuration, real model weights checkpoint selectors,
    a unified scrollable chat window with centered system analytical logs, and active
    question/interaction logging to the growing Master Corpus.

  PANEL B — Corpus Replay
    Load an existing v31 JSONL corpus, replay all records through
    inference engine, display per-record results, export replay log.

  PANEL C — Telemetry & Sync Center
    Live read of phase_gate_v31.jsonl. GCS secure upload center.
    Gate open/building ratio. V series over gate checks.

  PANEL D — Live Screen Recorder
    FFmpeg capture of host screen for documentation and presentation.
"""

from nicegui import ui
from stage3.inference_engine.main import V31InferenceEngine
from stage3.inference_engine.neural_inference import V31NeuralInferenceEngine
from stage2.dag_compiler.phase_gate import CONVERGENCE_ACTIONS
from stage1.core.triangulation import ACTION_VOCAB
from stage3.master_corpus.manager import MasterCorpusManager
from telemetry.gcs_sync import GCSSyncManager

import json
import time
import math
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
TELEMETRY_DIR = ROOT_DIR / "telemetry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── module-level state ─────────────────────────────────────────────────────────
_classical_engine: V31InferenceEngine | None = None
_neural_engine: V31NeuralInferenceEngine | None = None
_infer_history: list[dict] = []
_replay_results: list[dict] = []
_last_record_mode_index = 0
chat_messages: list[dict] = []


def _get_active_engine(is_neural: bool, l1_path: Optional[str] = None, l2_path: Optional[str] = None) -> V31InferenceEngine | V31NeuralInferenceEngine:
    global _classical_engine, _neural_engine
    if is_neural:
        if _neural_engine is None:
            _neural_engine = V31NeuralInferenceEngine(checkpoint_path="models/v31_neural_model.pt")
        
        # Soft load selected Layer 1 weights
        if l1_path == "simulated" or not l1_path:
            _neural_engine.neural_l1.is_loaded = False
        else:
            _neural_engine.neural_l1.load_weights(l1_path)
            
        # Soft load selected Layer 2 weights
        if l2_path:
            _neural_engine.load_weights(l2_path)
        else:
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


def scan_checkpoints() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Scans the Pt's/ and models/ directories for available Layer 1 and 2 PyTorch checkpoints."""
    l1_files = []
    l2_files = []
    
    search_dirs = [
        ROOT_DIR.parent / "Pt's",
        ROOT_DIR / "Pt's",
        ROOT_DIR / "models"
    ]
    
    seen_paths = set()
    
    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue
        try:
            for p in d.glob("*.pt"):
                abs_path = str(p.resolve())
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)
                
                name = p.name
                # Categorize based on prefix or content signature
                if name.startswith("layer1_") or name.startswith("stone_core_"):
                    l1_files.append({"label": f"L1: {name}", "value": abs_path, "name": name})
                elif name.startswith("l2_") or name == "v31_neural_model.pt" or "neural" in name:
                    l2_files.append({"label": f"L2: {name}", "value": abs_path, "name": name})
                else:
                    # File size classification fallback (L1 weight files are very tiny, <100KB)
                    if p.stat().st_size < 200_000:
                        l1_files.append({"label": f"L1: {name}", "value": abs_path, "name": name})
                    else:
                        l2_files.append({"label": f"L2: {name}", "value": abs_path, "name": name})
        except Exception as e:
            print(f"Error scanning checkpoints: {e}")
            
    # Sort files alphabetically
    l1_files.sort(key=lambda x: x["name"])
    l2_files.sort(key=lambda x: x["name"])
    
    # Prepend default option
    l1_files.insert(0, {"label": "Simulated Rule-Based (L1 Fallback)", "value": "simulated", "name": "simulated"})
    
    return l1_files, l2_files


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
                # Glowing blue-slate academic scale
                red = int(val_clamped * 0.18)
                green = int(val_clamped * 0.35)
                blue = int(val_clamped * 0.68)
            else:
                # Neutral slate-gray monochrome scale
                red = int(val_clamped * 0.45)
                green = int(val_clamped * 0.48)
                blue = int(val_clamped * 0.52)
                
            rects.append(
                f'<rect x="{c}" y="{r}" width="0.9" height="0.9" rx="0.15" '
                f'fill="rgb({red},{green},{blue})" />'
            )
            
    svg_body = "\n".join(rects)
    return (
        f'<svg viewBox="0 0 16 14" width="100%" height="100%" '
        f'style="background-color: #0f172a; border-radius: 4px; padding: 2px;">\n'
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
    return "text-emerald-600" if status == "OPEN" else "text-slate-500"


def stage3_view() -> None:
    # Scan checkpoints initially
    l1_files, l2_files = scan_checkpoints()
    
    # Establish default paths
    default_l1 = "simulated"
    for f in l1_files:
        if "layer1_combined" in f["name"]:
            default_l1 = f["value"]
            break
            
    default_l2 = ""
    for f in l2_files:
        if "v31_neural_model" in f["name"]:
            default_l2 = f["value"]
            break
    if not default_l2 and l2_files:
        default_l2 = l2_files[0]["value"]

    with ui.column().classes("w-full gap-4"):
        
        # Header Info (Simplified academic styling)
        with ui.row().classes("w-full items-center justify-between border-b border-slate-300 pb-2 mb-2"):
            with ui.column().classes("gap-0"):
                ui.label("Inference, Growing Corpus & Synchronization Center").classes("text-xl font-semibold text-slate-800")
                ui.label("Multi-cycle neural state progression with GCS storage backends and automated SOC 2/3 compliance pipeline.").classes("text-xs text-slate-500")

        # ══════════════════════════════════════════════════════════════════════
        # PANEL A — Unified Inference Arena
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-slate-300 p-4"):
            with ui.row().classes("w-full justify-between items-center mb-2 pb-2 border-b border-slate-200"):
                ui.label("🔬 Unified Inference Arena").classes("font-semibold text-slate-800 text-sm")
                engine_mode_toggle = ui.toggle({False: "Analytical ⚪", True: "Neural (PyTorch) 🟣"}, value=False).props("unelevated toggle-color=slate text-xs")

            # Model checkpoints dropdown selectors
            with ui.row().classes("w-full gap-3 items-center mb-3 flex-wrap"):
                l1_sel = ui.select(
                    label="Layer 1 Weights (M1, M2, M3)",
                    options={f["value"]: f["label"] for f in l1_files},
                    value=default_l1
                ).classes("flex-1 min-w-[200px]")
                
                l2_sel = ui.select(
                    label="Layer 2 Weights (Main Neural Model)",
                    options={f["value"]: f["label"] for f in l2_files},
                    value=default_l2
                ).classes("flex-1 min-w-[200px]")

            # Collapsible context inputs
            with ui.expansion("⚙ Configure Context Inputs", icon="settings").classes("w-full border border-slate-300 rounded bg-slate-50 mb-3"):
                with ui.column().classes("w-full p-3 gap-2 bg-transparent"):
                    infer_binary = ui.textarea(
                        label="Binary Context Text",
                        value="Phase gate logical: hash=v31all, mission=NextAura ATP inference, score=0.891"
                    ).classes("w-full h-16 text-xs")
                    infer_geometry = ui.textarea(
                        label="Geometry Context Text",
                        value="Entry 1. Env=Sandbox. SystemID=CASEBELIZE. TelemetryStable=True."
                    ).classes("w-full h-16 text-xs")
                    infer_stim = ui.number(label="Stimulus Scale Factor", value=1.0, min=0.1, max=10.0, step=0.1).classes("w-36")

            # Scrollable chat logs area
            chat_scroll = ui.scroll_area().classes("w-full h-96 bg-slate-950 rounded-lg p-3 border border-slate-300 shadow-inner")
            chat_container_element = [None]
            with chat_scroll:
                chat_container = ui.column().classes("w-full gap-2")
                chat_container_element[0] = chat_container

            # Unified Chat controls & toggles row
            with ui.row().classes("w-full items-center gap-3 mt-3 flex-wrap"):
                chat_input = ui.input(placeholder="Ask the model or send a sequence query (e.g. 'Synthesize system vectors', 'Evaluate telemetry matrix')...").classes("flex-1 text-xs")
                save_to_corpus_chk = ui.checkbox("Save to Master Corpus").classes("text-xs text-slate-600 font-semibold")
                send_btn = ui.button(icon="send").props("color=slate flat")
                reset_btn = ui.button("Reset Engine", on_click=lambda: _reset()).props("dense outline color=slate text-xs")

            # Inline log helper
            def _log_system_trace(text: str):
                """Logs analytical system traces directly into the chat flow as slate terminal messages."""
                chat_messages.append({
                    "sender": "system",
                    "text": text,
                    "timestamp": time.time()
                })
                _update_chat_ui()

            # Change triggers for dropdowns
            def on_l1_change(e):
                is_n = bool(engine_mode_toggle.value)
                eng = _get_active_engine(is_n, l1_path=l1_sel.value, l2_path=l2_sel.value)
                val = e.value
                if val == "simulated":
                    _log_system_trace("⚙️ Layer 1 estimator switched to Simulated Rule-Based mode.")
                else:
                    _log_system_trace(f"🧬 Switched Layer-1 Weights: {Path(val).name}")

            def on_l2_change(e):
                is_n = bool(engine_mode_toggle.value)
                eng = _get_active_engine(is_n, l1_path=l1_sel.value, l2_path=l2_sel.value)
                val = e.value
                if val:
                    _log_system_trace(f"🧠 Switched Layer-2 Model Weights: {Path(val).name}")

            l1_sel.on("change", on_l1_change)
            l2_sel.on("change", on_l2_change)

            # Output field indicators (Compact, clean row)
            ui.label("Observer Metrics Readout:").classes("text-[11px] font-semibold text-slate-500 mt-2")
            with ui.row().classes("gap-2 flex-wrap w-full mt-1"):
                out_fields: dict[str, ui.label] = {}
                for key in ["turn", "V", "gate_status", "M1", "M2", "M3",
                            "binary_scalar", "geometry_scalar", "language_scalar"]:
                    with ui.card().classes("p-2 flex-1 min-w-[100px] border border-slate-200 bg-slate-50 shadow-none gap-0 items-center justify-center"):
                        ui.label(key).classes("text-[9px] uppercase tracking-wider text-slate-400 font-bold")
                        lbl = ui.label("—").classes("text-xs font-mono font-bold text-slate-700")
                        out_fields[key] = lbl

            # Visual prediction frames and text synthesis outputs
            with ui.row().classes("gap-3 w-full mt-2 items-stretch flex-wrap"):
                nfp_card = ui.card().classes("flex-1 min-w-[220px] p-2.5 border border-slate-200 shadow-none")
                with nfp_card:
                    ui.label("next_frame_prediction").classes("text-[9px] uppercase font-bold text-slate-400 tracking-wider")
                    nfp_lbl = ui.label("—").classes("text-xs font-mono text-slate-600 mt-1")
                
                canvas_card = ui.card().classes("w-44 p-2.5 items-center justify-center bg-slate-950 border border-slate-800 shadow-none")
                with canvas_card:
                    ui.label("synthesized_visual_frame").classes("text-[9px] text-slate-400 mb-1 font-mono uppercase tracking-wider")
                    visual_grid_html = ui.html(generate_svg_grid([0.0]*224, is_neural=False)).classes("w-36 h-30")
                
                lto_card = ui.card().classes("flex-1 min-w-[220px] p-2.5 border border-slate-200 shadow-none")
                with lto_card:
                    ui.label("language_token_output").classes("text-[9px] uppercase font-bold text-slate-400 tracking-wider")
                    lto_lbl = ui.label("—").classes("text-base font-mono font-bold text-slate-700 mt-1")

            # V trajectory chart
            ui.label("Running V(s) Sequence Trajectory:").classes("text-[11px] font-semibold text-slate-500 mt-2")
            v_chart = ui.echart({
                "animation": False,
                "grid": {"top": 6, "bottom": 24, "left": 36, "right": 8},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 9}},
                "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"fontSize": 9}},
                "series": [{
                    "type": "line", "data": [], "smooth": True, "symbol": "circle", "symbolSize": 4,
                    "lineStyle": {"color": "#475569", "width": 2},
                    "areaStyle": {"color": "rgba(71,85,105,0.08)"},
                }],
            }).classes("w-full h-28 mt-1")

            def _update_chat_ui():
                container = chat_container_element[0]
                if not container:
                    return
                container.clear()
                with container:
                    if not chat_messages:
                        ui.label("No active chat. Type a message below to begin interaction trace.").classes(
                            "text-xs text-slate-400 italic text-center w-full my-4"
                        )
                        return
                        
                    for msg in chat_messages:
                        if msg["sender"] == "user":
                            # User bubble
                            with ui.row().classes("w-full justify-end gap-2 my-1"):
                                with ui.card().classes("bg-slate-200 text-slate-800 p-2.5 rounded-2xl rounded-tr-none max-w-[80%] shadow-none border border-slate-300"):
                                    ui.label(msg["text"]).classes("text-xs leading-relaxed break-words font-medium")
                                    ui.label("User").classes("text-[9px] text-slate-500 mt-0.5 text-right block font-semibold")
                        elif msg["sender"] == "system":
                            # System trace bubble
                            with ui.row().classes("w-full justify-center my-1"):
                                with ui.card().classes("bg-slate-900 border border-slate-700 text-slate-300 p-2 rounded-lg max-w-[95%] shadow shadow-slate-950"):
                                    ui.label(msg["text"]).classes("text-[10px] font-mono leading-relaxed whitespace-pre-wrap break-all")
                        else:
                            # Model response bubble
                            with ui.row().classes("w-full justify-start gap-2 my-1"):
                                bg_class = "bg-white border border-slate-300 text-slate-800" if "is_error" not in msg else "bg-red-50 border border-red-300 text-red-800"
                                with ui.card().classes(f"{bg_class} p-2.5 rounded-2xl rounded-tl-none max-w-[85%] shadow-none"):
                                    if "mode" in msg:
                                        with ui.row().classes("items-center justify-between w-full mb-1 border-b border-slate-100 pb-1 gap-4"):
                                            ui.label(msg["mode"]).classes("text-[9px] uppercase font-bold text-slate-500 tracking-wider")
                                            ui.label(f"Turn {msg['turn']}").classes("text-[9px] text-slate-400 font-mono")
                                    
                                    ui.label(msg["text"]).classes("text-xs font-semibold leading-relaxed break-words text-slate-800")
                                    
                                    if "v" in msg:
                                        with ui.expansion("Thought Trace & Frame Map", icon="analytics").classes("w-full text-[11px] text-slate-600 mt-1.5 bg-slate-50 rounded p-1 border border-slate-200"):
                                            with ui.row().classes("w-full gap-2 justify-between mb-1 pb-1 border-b border-slate-200"):
                                                with ui.column().classes("gap-0.5"):
                                                    ui.label(f"V(s) Triangulation: {msg['v']:.4f}").classes("font-mono text-slate-700 font-bold")
                                                    ui.label(f"Phase Gate Status: {msg['gate_status']}").classes(f"font-mono font-bold {'text-emerald-600' if msg['gate_status'] == 'OPEN' else 'text-slate-500'}")
                                                with ui.column().classes("gap-0.5 items-end"):
                                                    ui.label(f"M1={msg['m1']:.2f} | M2={msg['m2']:.2f}").classes("font-mono text-slate-500")
                                                    ui.label(f"M3={msg['m3']:.2f} | L1={msg['l1_checkpoint']}").classes("font-mono text-slate-500")
                                            
                                            ui.label("next_frame_prediction:").classes("text-[9px] text-slate-400 uppercase font-bold tracking-wider")
                                            ui.label(msg["nfp"]).classes("text-[10px] font-mono text-slate-600 bg-slate-100 p-1.5 rounded mb-1.5 leading-tight")
                                            
                                            ui.label("visual_frame_rendering:").classes("text-[9px] text-slate-400 uppercase font-bold tracking-wider mb-1")
                                            ui.html(generate_svg_grid(msg["pixels"], is_neural="Neural" in msg["mode"])).classes("w-28 h-24 mx-auto mb-1")
                                            
                                    ui.label("Model").classes("text-[9px] text-slate-500 mt-0.5 text-left block font-bold")
                
                # Auto-scroll container to bottom
                ui.run_javascript(f"var el = document.getElementById('{container.id}'); if(el) {{ el.scrollTop = el.scrollHeight; }}")

            async def _send_chat_message(user_msg_input):
                msg_text = user_msg_input.value.strip()
                if not msg_text:
                    return
                user_msg_input.set_value("")
                
                chat_messages.append({"sender": "user", "text": msg_text, "timestamp": time.time()})
                _update_chat_ui()
                
                is_n = bool(engine_mode_toggle.value)
                
                # Load selected checkpoints
                eng = _get_active_engine(is_n, l1_path=l1_sel.value, l2_path=l2_sel.value)
                
                _log_system_trace(f"[Inference Arena] Running model evaluation on turn {eng._turn + 1}...\n"
                                  f"├─ Input language_text: \"{msg_text}\"\n"
                                  f"├─ Active engine: {'Neural Model' if is_n else 'Analytical Fallback'}\n"
                                  f"└─ L1 weight file: {l1_sel.value if is_n else 'simulated'}")
                try:
                    result = eng.infer(
                        binary_text=infer_binary.value,
                        geometry_text=infer_geometry.value,
                        language_text=msg_text,
                        stimulus=float(infer_stim.value),
                    )
                    _infer_history.append(result)
                    
                    # Update Panel A's indicators
                    for key, lbl in out_fields.items():
                        val = result.get(key, "—")
                        lbl.set_text(str(val))
                        if key == "gate_status":
                            lbl.classes(remove="text-emerald-600 text-slate-500")
                            lbl.classes(add=_gate_color(str(val)))
                    
                    nfp_lbl.set_text(result.get("next_frame_prediction", "—"))
                    lto_lbl.set_text(result.get("language_token_output", "—"))
                    
                    # Update grid
                    if is_n and "neural_frame" in result:
                        grid_pixels = result["neural_frame"]
                    else:
                        grid_pixels = generate_classical_pattern(result["V"])
                    visual_grid_html.set_content(generate_svg_grid(grid_pixels, is_neural=is_n))
                    
                    # Update charts
                    v_vals = [r["V"] for r in _infer_history[-30:]]
                    v_chart.options["xAxis"]["data"] = [str(i+1) for i in range(len(v_vals))]
                    v_chart.options["series"][0]["data"] = v_vals
                    v_chart.update()
                    
                    # Log activity to traces
                    gate = result["gate_status"]
                    mode_lbl = "NEURAL" if is_n else "ANALYTICAL"
                    
                    _log_system_trace(f"[Inference Result] Complete evaluation successful.\n"
                                      f"├─ V-Value predicted: {result['V']:.5f}\n"
                                      f"├─ Extracted parameters: M1={result['M1']:.3f} | M2={result['M2']:.3f} | M3={result['M3']:.3f}\n"
                                      f"├─ Logical coherence gate: {gate}\n"
                                      f"└─ Synthesized action: \"{result['language_token_output']}\"")
                    
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
                        "l1_checkpoint": result.get("l1_checkpoint", "None"),
                        "turn": result.get("turn", 0),
                        "pixels": grid_pixels,
                        "timestamp": time.time()
                    })
                    _update_chat_ui()
                    
                    # Active Question logging to Master Corpus
                    if save_to_corpus_chk.value:
                        m_vals = {"M1": result.get("M1", 0.0), "M2": result.get("M2", 0.0), "M3": result.get("M3", 0.0)}
                        l2_scalars = {
                            "binary_scalar": result.get("binary_scalar", 0.0),
                            "geometry_scalar": result.get("geometry_scalar", 0.0),
                            "language_scalar": result.get("language_scalar", 0.0)
                        }
                        
                        added = MasterCorpusManager.add_interaction(
                            question=msg_text,
                            answer=result.get("language_token_output", "—"),
                            m_vals=m_vals,
                            l2_scalars=l2_scalars,
                            v_classical=result.get("V", 0.0),
                            action=result.get("language_token_output", "converge"),
                            next_frame_prediction=result.get("next_frame_prediction", "")
                        )
                        if added:
                            _log_system_trace("💾 [MasterCorpus] Interaction successfully appended to output/master_corpus.jsonl.")
                        else:
                            _log_system_trace("⚠️ [MasterCorpus] Duplicate state hash or input sequence skipped.")
                        _update_corpus_stats()
                except Exception as ex:
                    chat_messages.append({"sender": "model", "text": f"Error running inference: {ex}", "is_error": True})
                    _update_chat_ui()
                    _log_system_trace(f"❌ [Inference Error] Trace evaluation crash: {ex}")

            # Wire up chat triggers
            chat_input.on("keydown.enter", lambda: _send_chat_message(chat_input))
            send_btn.on("click", lambda: _send_chat_message(chat_input))
            _update_chat_ui()

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
                _log_system_trace("↺ Inference state and local context registers successfully reset.")

            def _export_infer():
                if not _infer_history:
                    _log_system_trace("⚠️ No inference history to export.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"inference_log_{ts}.jsonl"
                out.write_text("\n".join(json.dumps(r) for r in _infer_history))
                _log_system_trace(f"✅ Exported {len(_infer_history)} inference trials → {out.name}")

            def _export_pytorch_model():
                try:
                    import shutil
                    src = ROOT_DIR / "models" / "v31_neural_model.pt"
                    if not src.exists():
                        _log_system_trace("✗ Export PyTorch Error: v31_neural_model.pt not found. Run Stage 2 training first.")
                        return
                    ts = int(time.time())
                    dst = OUTPUT_DIR / f"model_export_v31_{ts}.pt"
                    shutil.copy(src, dst)
                    _log_system_trace(f"✅ Copied neural weights model to output archive → {dst.name}")
                except Exception as ex:
                    _log_system_trace(f"✗ Export PyTorch Error: {ex}")

            # Top controls footer
            with ui.row().classes("gap-2 mt-2"):
                ui.button("Export Inference Log", on_click=_export_infer).props("dense outline color=slate text-xs")
                ui.button("Export Active Weights (.pt)", on_click=_export_pytorch_model).props("dense outline color=slate text-xs")

        # ══════════════════════════════════════════════════════════════════════
        # MASTER CORPUS AUTO-GROW DROPZONE CARD
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-slate-300 p-4"):
            ui.label("📁 Master Corpus Auto-Grow Dropzone").classes("font-semibold text-slate-800 text-sm")
            ui.label(
                "Ingest raw datasets (JSON, JSONL, CSV) directly to the Master Corpus bucket. Records are deduplicated based on scalar hashes and secured via SOC 2/3 provenance blocks."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("w-full gap-2 items-center flex-wrap"):
                upload_zone = ui.upload(on_upload=lambda e: handle_corpus_upload(e)).classes("flex-1 min-w-[280px]")
                upload_zone.props("accept=.json,.jsonl,.csv max-file-size=52428800 label='Ingest file data'")
                
                with ui.column().classes("gap-2 min-w-[200px]"):
                    regen_btn = ui.button("Regenerate Corpus Outputs", on_click=lambda: _run_corpus_regeneration()).props("dense color=slate")
                    corpus_stats_lbl = ui.label("Querying Master Corpus statistics...").classes("text-xs font-mono text-slate-600 font-semibold")

            def handle_corpus_upload(e):
                try:
                    content_bytes = e.content.read()
                    filename = e.name
                    
                    # Process uploaded record file
                    res = MasterCorpusManager.add_file_to_master_corpus(content_bytes, filename)
                    
                    if "error" in res:
                        ui.notify(f"Ingestion failed: {res['error']}", type="negative")
                        _log_system_trace(f"❌ [MasterCorpus] Ingestion error: {res['error']}")
                    else:
                        _log_system_trace(f"📁 [MasterCorpus] Dataset: {res['filename']} ingested successfully!\n"
                                          f"├─ Total records parsed: {res['total_read']}\n"
                                          f"├─ Deduplicated clean additions: {res['added_count']}\n"
                                          f"├─ Skipped duplicates: {res['duplicates_skipped']}\n"
                                          f"└─ Corpus Hash Signature: {res.get('corpus_sha256_signature', '—')[:16]}...")
                        
                        ui.notify(f"Ingestion complete: +{res['added_count']} unique records!", type="positive")
                        _update_corpus_stats()
                except Exception as ex:
                    ui.notify(f"Upload exception: {ex}", type="negative")
                    _log_system_trace(f"❌ [MasterCorpus] Upload crash: {ex}")

            def _update_corpus_stats():
                path = ROOT_DIR / "output" / "master_corpus.jsonl"
                if not path.exists():
                    corpus_stats_lbl.set_text("Master Corpus Status: 0 records.")
                    return
                try:
                    count = 0
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                count += 1
                    corpus_stats_lbl.set_text(f"Master Corpus Status: {count} unique records saved.")
                except Exception as e:
                    corpus_stats_lbl.set_text(f"Master Corpus: read error ({e})")

            async def _run_corpus_regeneration():
                is_n = bool(engine_mode_toggle.value)
                eng = _get_active_engine(is_n, l1_path=l1_sel.value, l2_path=l2_sel.value)
                
                _log_system_trace("▶️ Starting Master Corpus Regeneration against the active network weights...")
                _log_system_trace(f"   ├─ Engine Type: {'Neural Model 🟣' if is_n else 'Analytical Fallback ⚪'}")
                if is_n:
                    _log_system_trace(f"   └─ Weights Checkpoint: {Path(eng.checkpoint_path).name}")
                    
                regen_btn.disable()
                try:
                    # Run CPU/GPU inference loop in background thread to keep UI alive
                    loop = asyncio.get_event_loop()
                    res = await loop.run_in_executor(None, MasterCorpusManager.regenerate_corpus, eng)
                    
                    if "error" in res:
                        _log_system_trace(f"❌ [MasterCorpus] Regeneration failed: {res['error']}")
                        ui.notify(f"Regeneration failed: {res['error']}", type="negative")
                    else:
                        _log_system_trace(f"✅ [MasterCorpus] Regeneration completed successfully!\n"
                                          f"├─ Evaluated and overwritten records: {res['regenerated_count']}\n"
                                          f"└─ Time elapsed: {res['elapsed_seconds']} seconds.")
                        ui.notify(f"Regenerated {res['regenerated_count']} records!", type="positive")
                        _update_corpus_stats()
                except Exception as e:
                    _log_system_trace(f"❌ [MasterCorpus] Regeneration crash: {e}")
                    ui.notify(f"Regeneration error: {e}", type="negative")
                finally:
                    regen_btn.enable()

            # Load initial stats
            _update_corpus_stats()

        # ══════════════════════════════════════════════════════════════════════
        # PANEL B — Corpus Replay (Clean Slate Styling)
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-slate-300 p-4"):
            with ui.row().classes("w-full justify-between items-center mb-1"):
                ui.label("▶▶ Corpus Replay").classes("font-semibold text-slate-800 text-sm")
                replay_mode_toggle = ui.toggle({False: "Analytical ⚪", True: "Neural (PyTorch) 🟣"}, value=False).props("unelevated toggle-color=slate text-xs")
            
            ui.label(
                "Evaluate a stored v31 JSONL corpus file record-by-record through the stateful inference pipelines."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3 items-end w-full"):
                replay_path_input = ui.input(
                    label="Corpus JSONL local path",
                    value="corpus_v31_sample.jsonl",
                ).classes("flex-1 text-xs")
                replay_btn = ui.button("Replay (Instant)").props("dense color=slate text-xs")
                replay_live_btn = ui.button("Replay (Live Animation)").props("dense outline color=slate text-xs")

            # Replay display canvas grid
            with ui.row().classes("w-full mt-2 items-center justify-center"):
                replay_canvas_card = ui.card().classes("w-44 p-2 items-center justify-center bg-slate-950 border border-slate-800 shadow-none")
                with replay_canvas_card:
                    ui.label("replay_visual_frame").classes("text-[9px] text-slate-400 mb-1 font-mono uppercase tracking-wider")
                    replay_grid_html = ui.html(generate_svg_grid([0.0]*224, is_neural=False)).classes("w-36 h-30")

            # Replay Stats Indicators
            with ui.row().classes("gap-4 mt-1 w-full justify-center"):
                replay_count_lbl = ui.label("records: —").classes("text-xs font-mono")
                replay_vmean_lbl = ui.label("V_mean: —").classes("text-xs font-mono")
                replay_gate_lbl = ui.label("gate OPEN: —").classes("text-xs font-mono text-slate-600")
                replay_lto_lbl = ui.label("top action: —").classes("text-xs font-mono text-slate-600")

            # Replay Chart
            replay_v_chart = ui.echart({
                "animation": False,
                "grid": {"top": 6, "bottom": 24, "left": 36, "right": 8},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 9}},
                "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"fontSize": 9}},
                "series": [
                    {
                        "type": "line", "data": [], "smooth": True, "symbol": "none",
                        "lineStyle": {"color": "#64748b", "width": 2},
                        "areaStyle": {"color": "rgba(100,116,139,0.08)"},
                        "name": "V",
                    }
                ],
            }).classes("w-full h-28 mt-1")

            replay_log = ui.log(max_lines=12).classes(
                "w-full text-xs font-mono h-24 bg-slate-900 text-slate-300 mt-2 border border-slate-700 rounded p-1"
            )

            def _resolve_replay_path() -> Path:
                path_str = replay_path_input.value.strip()
                path = Path(path_str)
                if not path.exists():
                    alt = ROOT_DIR / path_str
                    if alt.exists():
                        path = alt
                    else:
                        alt_out = OUTPUT_DIR / Path(path_str).name
                        if alt_out.exists():
                            path = alt_out
                return path

            def _replay_corpus():
                global _replay_results
                path = _resolve_replay_path()
                if not path.exists():
                    replay_log.push(f"✗ Replay source file not found: {replay_path_input.value}")
                    return

                is_n = bool(replay_mode_toggle.value)
                replay_log.push(f"▶ Replaying {path.name} using {'Neural' if is_n else 'Classical'} engine...")
                
                try:
                    records = []
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                continue
                    
                    if not records:
                        replay_log.push("✗ Corpus target contains no valid records.")
                        return

                    total = len(records)
                    eng = _get_active_engine(is_n, l1_path=l1_sel.value, l2_path=l2_sel.value)
                    eng.reset()

                    results = []
                    v_vals = []
                    for idx, rec in enumerate(records):
                        si = rec.get("scalar_inputs", {})
                        binary_text = si.get("binary", "") or str(rec.get("M1", ""))
                        geometry_text = si.get("geometry", "") or str(rec.get("M2", ""))
                        language_text = si.get("language", "") or str(rec.get("M3", ""))

                        if len(geometry_text.strip()) < 20:
                            geometry_text = f"Entry {idx+1}/{total}. Env=Sandbox. SystemID=CASEBELIZE. TelemetryStable=True. {geometry_text}".strip()

                        res = eng.infer(
                            binary_text=binary_text,
                            geometry_text=geometry_text,
                            language_text=language_text,
                            stimulus=float(rec.get("stimulus", 1.0)),
                            entry_index=idx + 1,
                            total_entries=total,
                        )
                        results.append(res)
                        v_vals.append(res["V"])

                    _replay_results = results
                    v_mean = sum(v_vals) / len(v_vals) if v_vals else 0.0
                    open_ct = sum(1 for r in results if r["gate_status"] == "OPEN")
                    
                    actions = [r["language_token_output"] for r in results]
                    act_count = {}
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

                    if is_n and "neural_frame" in results[-1]:
                        rep_pixels = results[-1]["neural_frame"]
                    else:
                        rep_pixels = generate_classical_pattern(results[-1]["V"])
                    replay_grid_html.set_content(generate_svg_grid(rep_pixels, is_neural=is_n))

                    replay_log.push(f"✅ Instant Replay Completed: V_mean={v_mean:.4f} | Gate Open ratio={open_ct}/{total}")
                except Exception as ex:
                    replay_log.push(f"✗ Replay exception: {ex}")

            async def _replay_corpus_live():
                global _replay_results
                path = _resolve_replay_path()
                if not path.exists():
                    replay_log.push(f"✗ Replay source file not found: {replay_path_input.value}")
                    return

                is_n = bool(replay_mode_toggle.value)
                replay_log.push(f"▶ [LIVE] Replaying {path.name}...")
                
                try:
                    records = []
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                continue

                    if not records:
                        replay_log.push("✗ Corpus is empty.")
                        return

                    total = len(records)
                    eng = _get_active_engine(is_n, l1_path=l1_sel.value, l2_path=l2_sel.value)
                    eng.reset()

                    _replay_results = []
                    v_vals = []

                    for idx, rec in enumerate(records):
                        si = rec.get("scalar_inputs", {})
                        binary_text = si.get("binary", "") or str(rec.get("M1", ""))
                        geometry_text = si.get("geometry", "") or str(rec.get("M2", ""))
                        language_text = si.get("language", "") or str(rec.get("M3", ""))

                        if len(geometry_text.strip()) < 20:
                            geometry_text = f"Entry {idx+1}/{total}. Env=Sandbox. SystemID=CASEBELIZE. {geometry_text}".strip()

                        result = eng.infer(
                            binary_text=binary_text,
                            geometry_text=geometry_text,
                            language_text=language_text,
                            stimulus=float(rec.get("stimulus", 1.0)),
                            entry_index=idx + 1,
                            total_entries=total,
                        )
                        _replay_results.append(result)
                        v_vals.append(result["V"])
                        v_mean = sum(v_vals) / len(v_vals)
                        open_ct = sum(1 for r in _replay_results if r["gate_status"] == "OPEN")

                        # Update UI
                        replay_count_lbl.set_text(f"records: {idx+1}/{total}")
                        replay_vmean_lbl.set_text(f"V_mean: {v_mean:.4f}")
                        replay_gate_lbl.set_text(f"gate OPEN: {open_ct}/{idx+1}")
                        replay_lto_lbl.set_text(f"action: {result['language_token_output']}")

                        if is_n and "neural_frame" in result:
                            rep_pixels = result["neural_frame"]
                        else:
                            rep_pixels = generate_classical_pattern(result["V"])
                        replay_grid_html.set_content(generate_svg_grid(rep_pixels, is_neural=is_n))

                        # Live chart updating
                        replay_v_chart.options["xAxis"]["data"] = [str(i+1) for i in range(len(v_vals))]
                        replay_v_chart.options["series"][0]["data"] = [round(v, 4) for v in v_vals]
                        replay_v_chart.update()

                        replay_log.push(f"[{idx+1}/{total}] V={result['V']:.4f} gate={result['gate_status']} action={result['language_token_output']}")
                        await asyncio.sleep(0.04)

                    replay_log.push("✅ Live Animated Replay completed.")
                except Exception as ex:
                    replay_log.push(f"✗ Live Replay error: {ex}")

            def _export_replay():
                if not _replay_results:
                    replay_log.push("No replay logs available to write.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"replay_results_{ts}.jsonl"
                slim = [{k: v for k, v in r.items() if k != "source_record"} for r in _replay_results]
                out.write_text("\n".join(json.dumps(r) for r in slim))
                replay_log.push(f"✅ Exported replay log → {out.name}")

            replay_btn.on("click", lambda: _replay_corpus())
            replay_live_btn.on("click", lambda: _replay_corpus_live())
            ui.button("Export Replay Results", on_click=_export_replay).props("dense outline color=slate text-xs")

        # ══════════════════════════════════════════════════════════════════════
        # PANEL C — Telemetry & GCS Sync Card
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-slate-300 p-4"):
            ui.label("📡 Telemetry Monitoring & GCS Sync").classes("font-semibold text-slate-800 text-sm")
            ui.label(
                "Monitor live system logs at telemetry/phase_gate_v31.jsonl and synchronize active state databases securely with Google Cloud Storage bucket."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-2 items-center flex-wrap"):
                telem_load_btn = ui.button("Refresh Gate Metrics").props("dense color=slate text-xs")
                telem_export_btn = ui.button("Export Snapshot").props("dense outline color=slate text-xs")

            # Summary metrics
            with ui.row().classes("gap-4 mt-2 w-full justify-center"):
                t_total_lbl = ui.label("checks: —").classes("text-xs font-mono")
                t_open_lbl = ui.label("OPEN: —").classes("text-xs font-mono text-emerald-600 font-semibold")
                t_building_lbl = ui.label("BUILDING: —").classes("text-xs font-mono text-slate-600 font-semibold")
                t_ratio_lbl = ui.label("open ratio: —").classes("text-xs font-mono")

            # Condition sliders
            ui.label("Condition verification rates:").classes("text-[10px] text-slate-400 mt-2 font-mono uppercase tracking-wider")
            cond_bars: dict[str, tuple] = {}
            for cond in ["v_ok", "action_ok", "m2_ok", "m3_ok"]:
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(cond).classes("w-20 text-xs font-mono")
                    bar_wrap = ui.element("div").classes("flex-1 bg-slate-100 rounded h-3 overflow-hidden")
                    with bar_wrap:
                        bar = ui.element("div").classes("bg-slate-500 h-full rounded").style("width: 0%")
                    rate_lbl = ui.label("—").classes("w-12 text-xs font-mono text-right")
                    cond_bars[cond] = (bar, rate_lbl)

            # ECharts gate series
            telem_v_chart = ui.echart({
                "animation": False,
                "grid": {"top": 6, "bottom": 24, "left": 36, "right": 8},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 9}},
                "yAxis": {"type": "value", "min": 0, "max": 1, "axisLabel": {"fontSize": 9}},
                "series": [
                    {
                        "type": "line", "data": [], "smooth": False, "symbol": "circle", "symbolSize": 3,
                        "lineStyle": {"color": "#475569", "width": 1.5},
                        "name": "V",
                    },
                    {
                        "type": "line", "data": [], "smooth": False, "symbol": "none",
                        "lineStyle": {"color": "#10b981", "width": 1, "type": "dashed"},
                        "name": "threshold (0.75)",
                    },
                ],
            }).classes("w-full h-28 mt-2")

            telem_log = ui.log(max_lines=14).classes(
                "w-full text-xs font-mono h-24 bg-slate-900 text-slate-300 mt-2 border border-slate-700 rounded p-1"
            )

            def _load_telemetry():
                path = TELEMETRY_DIR / "phase_gate_v31.jsonl"
                if not path.exists():
                    telem_log.push("No active gate log found. Evaluated trials in Panel A will generate telemetry.")
                    return
                try:
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

                    total = len(entries)
                    open_c = sum(1 for e in entries if e.get("status") == "OPEN")
                    build_c = total - open_c
                    ratio = open_c / total if total else 0.0

                    t_total_lbl.set_text(f"checks: {total}")
                    t_open_lbl.set_text(f"OPEN: {open_c}")
                    t_building_lbl.set_text(f"BUILDING: {build_c}")
                    t_ratio_lbl.set_text(f"open ratio: {ratio:.2%}")

                    # Condition hit rates
                    for cond in ["v_ok", "action_ok", "m2_ok", "m3_ok"]:
                        hit = sum(1 for e in entries if e.get(cond) is True)
                        rate = hit / total if total else 0.0
                        bar, lbl = cond_bars[cond]
                        bar.style(f"width: {rate*100:.1f}%")
                        lbl.set_text(f"{rate:.0%}")

                    # V series chart
                    recent = entries[-50:]
                    v_vals = [e.get("V", 0.0) for e in recent]
                    x_data = [str(i+1) for i in range(len(v_vals))]
                    telem_v_chart.options["xAxis"]["data"] = x_data
                    telem_v_chart.options["series"][0]["data"] = [round(v, 4) for v in v_vals]
                    telem_v_chart.options["series"][1]["data"] = [0.75] * len(v_vals)
                    telem_v_chart.update()

                    telem_log.push(f"Telemetry Refresh: {total} logs. OPEN={open_c} ({ratio:.1%}) | BUILDING={build_c}")
                except Exception as ex:
                    telem_log.push(f"Error compiling telemetry: {ex}")

            def _export_telem():
                path = TELEMETRY_DIR / "phase_gate_v31.jsonl"
                if not path.exists():
                    telem_log.push("No logs available to export.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"telemetry_snapshot_{ts}.jsonl"
                out.write_text(path.read_text())
                telem_log.push(f"✅ Telemetry snapshot saved → {out.name}")

            telem_load_btn.on("click", lambda: _load_telemetry())
            telem_export_btn.on("click", lambda: _export_telem())
            _load_telemetry()

            # Google Cloud Storage Sync Center Sub-Layout
            ui.separator().classes("my-3")
            ui.label("☁️ Google Cloud Storage Sync Center").classes("font-semibold text-slate-800 text-[13px]")
            ui.label(
                "Securely synchronize model weights, state registries, and the growing master corpus directly to GCS bucket."
            ).classes("text-[11px] text-slate-500 mb-1.5")

            with ui.row().classes("w-full gap-2 items-end flex-wrap"):
                gcs_bucket_input = ui.input(
                    label="GCS Bucket Destination",
                    value="nextaura-research-vault"
                ).classes("flex-1 text-xs min-w-[220px]")
                
                sync_btn = ui.button("Sync to GCS", on_click=lambda: _run_gcs_sync()).props("dense color=slate text-xs")

            # Logs panel for GCS sync
            ui.label("Sync Action Logs:").classes("text-[9px] text-slate-400 mt-1.5 font-mono uppercase tracking-wider")
            gcs_log = ui.log(max_lines=15).classes(
                "w-full text-xs font-mono h-24 bg-slate-900 text-slate-300 border border-slate-700 rounded p-1"
            )

            async def _run_gcs_sync():
                bucket = gcs_bucket_input.value.strip()
                if not bucket:
                    ui.notify("GCS bucket name cannot be empty.", type="warning")
                    return
                    
                sync_btn.disable()
                gcs_log.clear()
                gcs_log.push("[GCS-CLI] Spawning non-blocking background CLI thread...")
                _log_system_trace(f"☁️ [GCS] Starting synchronization pipeline to bucket: {bucket}")
                
                try:
                    # Run in non-blocking thread pool
                    loop = asyncio.get_event_loop()
                    success, logs = await loop.run_in_executor(None, GCSSyncManager.sync_to_gcs, bucket)
                    
                    for l in logs:
                        gcs_log.push(l)
                        # Also feed directly to unified interactive chat log
                        _log_system_trace(l)
                        
                    if success:
                        ui.notify("GCS sync completed successfully!", type="positive")
                    else:
                        ui.notify("Sync complete with some cautions/warnings.", type="warning")
                except Exception as e:
                    gcs_log.push(f"[GCS] Sync Exception: {e}")
                    _log_system_trace(f"❌ [GCS] Synchronization pipeline crash: {e}")
                    ui.notify(f"Sync failed: {e}", type="negative")
                finally:
                    sync_btn.enable()

        # ══════════════════════════════════════════════════════════════════════
        # PANEL D — Screen Recorder (Clean Slate Styling)
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-slate-300 p-4"):
            ui.label("🎥 Screen Recorder").classes("font-semibold text-slate-800 text-sm")
            ui.label(
                "Record host desktop interactions or the running NiceGUI tab using FFmpeg for documentation."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3 items-end w-full flex-wrap"):
                rec_duration = ui.number(label="Duration (sec)", value=3.0, min=1.0, max=60.0, step=1.0).classes("w-24")
                rec_fps = ui.number(label="Framerate (fps)", value=10.0, min=5.0, max=30.0, step=5.0).classes("w-24")
                rec_mode = ui.select(
                    label="Recording Target Scope", 
                    options=[
                        "Full Desktop 🖥", 
                        "Chrome Browser Only (1/3 Complexity) 🌐",
                        "Cycle Both Modes 🔀 (1,2,1,2...)"
                    ], 
                    value="Full Desktop 🖥"
                ).classes("w-72 text-xs")
                rec_btn = ui.button("Record Screen").props("dense color=slate text-xs")

            # status & video preview
            rec_status = ui.label("Ready").classes("text-xs font-mono text-slate-500 mt-1")
            
            video_container = ui.column().classes("w-full mt-2 hidden")
            with video_container:
                ui.label("Recorded video:").classes("text-xs text-slate-500 font-mono")
                video_player = ui.video("").classes("w-full h-48 bg-black rounded")
                download_link = ui.link("Download file (.mp4)", "#").classes("text-xs text-slate-600 font-semibold underline")

            async def _start_recording():
                duration = float(rec_duration.value)
                fps = int(rec_fps.value)
                ts = int(time.time())
                output_filename = f"screen_rec_{ts}.mp4"
                
                static_path = Path(__file__).resolve().parent.parent / "static"
                recordings_path = static_path / "recordings"
                recordings_path.mkdir(parents=True, exist_ok=True)
                
                output_file_wsl = recordings_path / output_filename
                
                # Convert path representation for ffmpeg on Windows host execution
                abs_wsl_path = str(output_file_wsl.resolve())
                if abs_wsl_path.startswith("/mnt/"):
                    drive = abs_wsl_path[5].upper()
                    win_path = f"{drive}:" + abs_wsl_path[6:]
                else:
                    win_path = abs_wsl_path.replace("\\", "/")
                
                rec_btn.disable()
                rec_status.set_text("🔴 INITIALIZING FFmpeg...")
                rec_status.classes(remove="text-slate-500 text-slate-600")
                rec_status.classes(add="text-red-600 font-bold")
                
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
                    
                    # Live ticking countdown
                    for sec in range(int(duration), 0, -1):
                        rec_status.set_text(f"🔴 RECORDING [{actual_mode_name}]... {sec}s remaining")
                        await asyncio.sleep(1.0)
                        
                    stdout, stderr = await proc.communicate()
                    returncode = proc.returncode
                    
                    # Capture window handle fallback if gdigrab window not found
                    if returncode != 0 and target_input != "desktop":
                        rec_status.set_text("⚠️ WINDOW NOT FOUND. FALLING BACK TO DESKTOP CAPTURE...")
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
                            rec_status.set_text(f"🔴 RECORDING DESKTOP FALLBACK... {sec}s remaining")
                            await asyncio.sleep(1.0)
                        stdout, stderr = await proc_fb.communicate()
                        returncode = proc_fb.returncode
                    
                    if returncode == 0 and output_file_wsl.exists():
                        rec_status.set_text(f"✓ Capture saved successfully: {output_filename}")
                        rec_status.classes(remove="text-red-600")
                        rec_status.classes(add="text-emerald-600 font-bold")
                        
                        video_container.classes(remove="hidden")
                        video_url = f"/static/recordings/{output_filename}"
                        video_player.set_source(video_url)
                        download_link.props(f'href="{video_url}" target="_blank"')
                        ui.notify(f"Screen recording complete! [{actual_mode_name}]", type="positive")
                    else:
                        err_msg = stderr.decode(errors="ignore").strip().split("\n")[-1]
                        rec_status.set_text(f"✗ Capture failed: {err_msg}")
                        rec_status.classes(remove="text-red-600 text-emerald-600")
                        rec_status.classes(add="text-red-600 font-bold")
                        ui.notify(f"FFmpeg failed: {err_msg}", type="negative")
                        
                except Exception as ex:
                    rec_status.set_text(f"✗ Capture exception: {ex}")
                    rec_status.classes(remove="text-red-600 text-emerald-600")
                    rec_status.classes(add="text-red-600 font-bold")
                    ui.notify(f"FFmpeg error: {ex}", type="negative")
                finally:
                    rec_btn.enable()

            rec_btn.on("click", lambda: _start_recording())
