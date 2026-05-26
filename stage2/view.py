"""gui/stage2/view.py — V31all Stage 2 GUI
Surfaces the actual stage2 backends:
  - PhaseGate: live V-gate status, 4-condition breakdown, history log
  - TriangulationLoss: compute and display all 4 loss components
  - CorpusGenerator: generate corpus entries from spec, stats readout
  - CorpusIngest: ingest JSONL, validate/coerce to v31 schema, export

All panels read from / write to the real stage2 modules.
"""

from nicegui import ui
from stage2.dag_compiler.phase_gate import PhaseGate, CONVERGENCE_ACTIONS
from stage2.training_engine.triangulation_loss import total_triangulation_loss
from stage2.training_engine.corpus_generator import CorpusGenerator, generate_default_specs
from stage1.core.m_scalars import M1State, M2State, M3State, compute_m_scalars
from stage1.core.triangulation import TriangulationState, triangulate, ACTION_VOCAB
import json, time, math
from pathlib import Path
import torch
import asyncio
from stage2.training_engine.neural_model import V31Model
from stage2.training_engine.asymmetric_trainer import AsymmetricTrainer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
TELEMETRY_DIR = ROOT_DIR / "telemetry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── module-level state ─────────────────────────────────────────────────────────
_gate = PhaseGate()
_gate_history: list[dict] = []
_loss_history: list[dict] = []
_v_history: list[float] = []
_gen_records: list[dict] = []


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _status_color(status: str) -> str:
    return "text-emerald-600 font-bold" if status == "OPEN" else "text-amber-600 font-bold"


def _loss_color(val: float) -> str:
    if val < 0.05:
        return "text-emerald-600"
    if val < 0.20:
        return "text-amber-500"
    return "text-rose-600"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 VIEW
# ─────────────────────────────────────────────────────────────────────────────
def stage2_view() -> None:
    with ui.column().classes("w-full gap-4"):

        ui.label("⚙ Stage 2 — Training Engine").classes("text-lg font-bold text-emerald-700")
        ui.label(
            "Phase gate, triangulation loss, corpus generation and ingestion."
        ).classes("text-xs text-slate-500")

        # ══════════════════════════════════════════════════════════════════════
        # PANEL A — PhaseGate
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-emerald-300 p-3"):
            ui.label("🚦 Phase Gate").classes("font-semibold text-emerald-700 text-sm")
            ui.label(
                f"Opens when: V ≥ 0.75, action ∈ CONVERGENCE, M2 > 0.20, M3 ∈ [0.70, 1.50] "
                f"for 2 consecutive cycles"
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3 items-end w-full"):
                gate_v_input    = ui.number(label="V(s)",    value=0.80, min=0.0, max=1.0, step=0.01).classes("w-20")
                gate_m2_input   = ui.number(label="M2",      value=0.35, min=0.0, max=1.0, step=0.01).classes("w-20")
                gate_m3_input   = ui.number(label="M3",      value=1.10, min=0.5, max=2.0, step=0.05).classes("w-20")
                actions = sorted(CONVERGENCE_ACTIONS) + [a for a in ACTION_VOCAB if a not in CONVERGENCE_ACTIONS]
                gate_action_sel = ui.select(label="Action", options=actions, value="converge").classes("w-32")
                gate_run_btn    = ui.button("Check Gate").props("dense color=emerald")

            # condition row
            with ui.row().classes("gap-3 mt-1"):
                v_cond    = ui.label("V ≥ 0.75 : —").classes("text-xs font-mono")
                a_cond    = ui.label("action   : —").classes("text-xs font-mono")
                m2_cond   = ui.label("M2 > 0.20: —").classes("text-xs font-mono")
                m3_cond   = ui.label("M3 ∈ range: —").classes("text-xs font-mono")

            with ui.row().classes("gap-4 mt-1 items-center"):
                gate_status_lbl  = ui.label("STATUS: —").classes("text-xl font-mono font-bold")
                gate_consec_lbl  = ui.label("consecutive: —").classes("text-xs text-slate-500")

            gate_log = ui.log(max_lines=10).classes(
                "w-full text-xs font-mono h-20 bg-slate-900 text-emerald-300 mt-1"
            )

            def _check_gate():
                V   = float(gate_v_input.value)
                M2  = float(gate_m2_input.value)
                M3  = float(gate_m3_input.value)
                act = gate_action_sel.value

                status = _gate.check(V=V, action=act, M2=M2, M3=M3)
                _gate_history.append({"V": V, "M2": M2, "M3": M3, "action": act, "status": status})

                # condition breakdown
                ok_v  = V >= 0.75;  ok_a = act in CONVERGENCE_ACTIONS
                ok_m2 = M2 > 0.20;  ok_m3 = 0.70 <= M3 <= 1.50

                def _tick(b): return "✅" if b else "❌"
                v_cond.set_text(f"V ≥ 0.75 : {_tick(ok_v)} ({V:.3f})")
                a_cond.set_text(f"action   : {_tick(ok_a)} ({act})")
                m2_cond.set_text(f"M2 > 0.20: {_tick(ok_m2)} ({M2:.3f})")
                m3_cond.set_text(f"M3 ∈ range: {_tick(ok_m3)} ({M3:.3f})")

                consec = sum(1 for e in _gate_history[-3:] if e["status"] == "OPEN")
                gate_status_lbl.set_text(f"STATUS: {status}")
                gate_status_lbl.classes(remove="text-emerald-600 text-amber-600")
                gate_status_lbl.classes(add=_status_color(status).split()[0])
                gate_consec_lbl.set_text(f"recent OPEN: {consec}/3 checks")
                gate_log.push(f"[{status}] V={V:.3f} M2={M2:.3f} M3={M3:.3f} act={act}")

            gate_run_btn.on("click", lambda: _check_gate())

            # read existing gate log
            def _load_gate_log():
                path = TELEMETRY_DIR / "phase_gate_v31.jsonl"
                if not path.exists():
                    gate_log.push("No gate log yet.")
                    return
                lines = path.read_text().strip().split("\n")
                for line in lines[-8:]:
                    try:
                        e = json.loads(line)
                        gate_log.push(
                            f"[{e['status']}] V={e['V']:.3f} M2={e['M2']:.3f} "
                            f"M3={e['M3']:.3f} act={e['action']}"
                        )
                    except Exception:
                        pass

            ui.button("Load Gate History", on_click=_load_gate_log).props("dense outline color=emerald")

        # ══════════════════════════════════════════════════════════════════════
        # PANEL B — Triangulation Loss
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-violet-300 p-3"):
            ui.label("📉 Triangulation Loss").classes("font-semibold text-violet-700 text-sm")
            ui.label(
                "4 components: V-prediction, action alignment, invariant self-consistency, M3 budget"
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3 items-end flex-wrap"):
                pred_v_input    = ui.number(label="Predicted V",    value=0.72, min=0.0, max=1.0, step=0.01).classes("w-24")
                target_r_input  = ui.number(label="Target Reward",  value=0.80, min=0.0, max=1.0, step=0.01).classes("w-28")
                pred_act_sel    = ui.select(label="Predicted Action", options=ACTION_VOCAB, value="converge").classes("w-32")
                expert_act_sel  = ui.select(label="Expert Action",    options=ACTION_VOCAB, value="deploy").classes("w-28")
                m3_mult_input   = ui.number(label="M3 Multiplier",   value=1.10, min=0.5, max=2.0, step=0.05).classes("w-28")
                elapsed_input   = ui.number(label="Elapsed Frac",    value=0.30, min=0.0, max=1.0, step=0.05).classes("w-24")
                loss_btn        = ui.button("Compute Loss").props("dense color=deep-purple")

            # loss component bars
            ui.label("Loss components (lower = better):").classes("text-xs text-slate-500 mt-2")
            loss_component_rows = {}
            for comp in ["total", "v_prediction", "action_alignment", "invariant", "m3_budget"]:
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(comp).classes("w-32 text-xs font-mono")
                    bar_wrap = ui.element("div").classes("flex-1 bg-slate-100 rounded h-3 overflow-hidden")
                    with bar_wrap:
                        bar = ui.element("div").classes("bg-violet-500 h-full rounded").style("width: 0%")
                    val_lbl = ui.label("—").classes("w-16 text-xs font-mono text-right")
                    loss_component_rows[comp] = (bar, val_lbl)

            loss_log = ui.log(max_lines=8).classes(
                "w-full text-xs font-mono h-20 bg-slate-900 text-violet-300 mt-1"
            )

            def _compute_loss():
                pv   = float(pred_v_input.value)
                tr   = float(target_r_input.value)
                pa   = pred_act_sel.value
                ea   = expert_act_sel.value
                m3   = float(m3_mult_input.value)
                elap = float(elapsed_input.value)

                losses = total_triangulation_loss(
                    predicted_v=pv,
                    target_reward=tr,
                    predicted_action=pa,
                    expert_action=ea,
                    action_vocab=ACTION_VOCAB,
                    v_history=_v_history[-10:] if _v_history else [pv],
                    m3_multiplier=m3,
                    elapsed_fraction=elap,
                )
                _loss_history.append(losses)

                # update bars — normalize against total for proportional display
                max_val = max(losses.values()) if max(losses.values()) > 1e-9 else 1.0
                for comp, (bar, lbl) in loss_component_rows.items():
                    val = losses[comp]
                    pct = (val / max_val) * 100.0
                    bar.style(f"width: {pct:.1f}%")
                    lbl.set_text(f"{val:.5f}")

                loss_log.push(
                    f"total={losses['total']:.5f} | "
                    f"v_pred={losses['v_prediction']:.5f} | "
                    f"align={losses['action_alignment']:.5f} | "
                    f"inv={losses['invariant']:.5f} | "
                    f"m3={losses['m3_budget']:.5f}"
                )

                def _export_losses():
                    ts = int(time.time())
                    out = OUTPUT_DIR / f"loss_history_{ts}.jsonl"
                    out.write_text("\n".join(json.dumps(l) for l in _loss_history))
                    loss_log.push(f"✅ Exported {len(_loss_history)} loss records → {out.name}")

            loss_btn.on("click", lambda: _compute_loss())
            ui.button("Export Loss History", on_click=lambda: _export_loss_direct()).props(
                "dense outline color=deep-purple"
            )

            def _export_loss_direct():
                if not _loss_history:
                    loss_log.push("No losses to export yet.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"loss_history_{ts}.jsonl"
                out.write_text("\n".join(json.dumps(l) for l in _loss_history))
                loss_log.push(f"✅ Exported {len(_loss_history)} records → {out.name}")

        # ══════════════════════════════════════════════════════════════════════
        # PANEL C — Corpus Generator
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-sky-300 p-3"):
            ui.label("🗄 Corpus Generator").classes("font-semibold text-sky-700 text-sm")
            ui.label(
                "Generate v31 JSONL corpus (7-input → 2-output schema) from default or custom specs."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-3 items-end"):
                gen_n_input     = ui.number(label="Entries", value=20, min=1, max=200, step=1).classes("w-20")
                gen_theme_input = ui.input(label="Theme", value="adapter_training").classes("w-40")
                gen_fname_input = ui.input(label="Output filename", value="corpus_generated").classes("w-44")
                gen_btn         = ui.button("Generate Corpus").props("dense color=sky")

            # stats row
            with ui.row().classes("gap-4 mt-1"):
                stat_total  = ui.label("records: —").classes("text-xs font-mono")
                stat_vmean  = ui.label("V_mean: —").classes("text-xs font-mono")
                stat_vmin   = ui.label("V_min: —").classes("text-xs font-mono")
                stat_vmax   = ui.label("V_max: —").classes("text-xs font-mono")

            # action distribution chart
            ui.label("Action distribution (last run):").classes("text-xs text-slate-500 mt-1")
            action_dist_chart = ui.echart({
                "animation": False,
                "grid": {"top": 4, "bottom": 24, "left": 40, "right": 8},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 8, "rotate": 30}},
                "yAxis": {"type": "value", "axisLabel": {"fontSize": 9}},
                "series": [{"type": "bar", "data": [], "itemStyle": {"color": "#0ea5e9"}}],
            }).classes("w-full h-28")

            gen_log = ui.log(max_lines=8).classes(
                "w-full text-xs font-mono h-20 bg-slate-900 text-sky-300 mt-1"
            )

            def _generate_corpus():
                global _gen_records
                n     = int(gen_n_input.value or 20)
                theme = gen_theme_input.value or "adapter_training"
                fname = gen_fname_input.value or "corpus_generated"
                ts    = int(time.time())
                out   = OUTPUT_DIR / f"{fname}_{ts}.jsonl"

                gen_log.push(f"▶ Generating {n} entries | theme={theme}...")
                try:
                    specs  = generate_default_specs(n=n, theme=theme)
                    generator = CorpusGenerator(
                        output_path=str(out),
                        total_entries=n,
                        codebase_hash="v31all",
                        mission=f"gui_run_{theme}",
                    )
                    _gen_records = generator.run(specs)
                    stats = generator.stats()

                    stat_total.set_text(f"records: {stats['total_records']}")
                    stat_vmean.set_text(f"V_mean: {stats['V_mean']}")
                    stat_vmin.set_text(f"V_min: {stats['V_min']}")
                    stat_vmax.set_text(f"V_max: {stats['V_max']}")

                    # update action dist chart
                    dist = stats.get("action_distribution", {})
                    sorted_dist = sorted(dist.items(), key=lambda x: -x[1])
                    action_dist_chart.options["xAxis"]["data"] = [k for k, _ in sorted_dist]
                    action_dist_chart.options["series"][0]["data"] = [v for _, v in sorted_dist]
                    action_dist_chart.update()

                    gen_log.push(f"✅ {n} records → {out.name}")
                    gen_log.push(f"   V_mean={stats['V_mean']} min={stats['V_min']} max={stats['V_max']}")
                    gen_log.push(f"   actions: {dist}")
                except Exception as ex:
                    gen_log.push(f"✗ Error: {ex}")

            gen_btn.on("click", lambda: _generate_corpus())

        # ══════════════════════════════════════════════════════════════════════
        # PANEL D — Corpus Ingest (v31 schema validator)
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-orange-300 p-3"):
            ui.label("📥 Corpus Ingest & Validator").classes("font-semibold text-orange-700 text-sm")
            ui.label(
                "Upload or paste JSONL → coerce to v31 schema → validate → export cleaned corpus."
            ).classes("text-xs text-slate-500 mb-2")

            with ui.row().classes("gap-2 w-full"):
                ingest_paste = ui.textarea(
                    label="Paste JSONL (one record per line)", placeholder='{"M1":"...","scalar_inputs":{...},...}'
                ).classes("flex-1 h-24")
                ui.button(
                    "Validate & Ingest", on_click=lambda: _ingest_paste(ingest_paste.value)
                ).props("dense color=orange")

            with ui.row().classes("gap-4 mt-1"):
                ingest_ok_lbl   = ui.label("valid: —").classes("text-xs font-mono text-emerald-600")
                ingest_skip_lbl = ui.label("skipped: —").classes("text-xs font-mono text-rose-600")
                ingest_total_lbl = ui.label("total: —").classes("text-xs font-mono")

            ingest_log = ui.log(max_lines=10).classes(
                "w-full text-xs font-mono h-20 bg-slate-900 text-orange-300 mt-1"
            )

            _ingest_buffer: list[dict] = []

            REQ_TOP    = ("M1", "M2", "M3", "scalar_inputs", "output_tokens")
            REQ_SCALAR = ("binary", "geometry", "language", "mathematical_relationship")
            REQ_OUT    = ("next_frame_prediction", "language_token_output")

            def _coerce(rec: dict, idx: int):
                """Minimal coerce matching corpus_ingest.py logic."""
                si = rec.get("scalar_inputs", {}) if isinstance(rec.get("scalar_inputs"), dict) else {}
                ot = rec.get("output_tokens", {}) if isinstance(rec.get("output_tokens"), dict) else {}
                lb = str(rec.get("Binary") or rec.get("binary") or si.get("binary") or "")
                lg = str(rec.get("Geometry") or rec.get("geometry") or si.get("geometry") or "")
                ll = str(rec.get("LanguageContext") or rec.get("language") or si.get("language") or "")
                lm = str(rec.get("MathematicalRelationshipTriangulation") or si.get("mathematical_relationship") or "")
                nfp = str(rec.get("NextFramePrediction") or ot.get("next_frame_prediction") or "")
                lto = str(rec.get("LanguageTokenOutput") or ot.get("language_token_output") or "")
                if not lto:
                    lto = "continue_monitor"
                if not nfp:
                    nfp = f"Ingested next frame from: {ll[:96]}"
                return {
                    "M1": str(rec.get("M1", f"row_{idx}")),
                    "M2": str(rec.get("M2", f"row_{idx}")),
                    "M3": str(rec.get("M3", f"row_{idx}")),
                    "scalar_inputs": {"binary": lb, "geometry": lg, "language": ll, "mathematical_relationship": lm},
                    "output_tokens": {"next_frame_prediction": nfp, "language_token_output": lto},
                }

            def _ingest_paste(text: str):
                _ingest_buffer.clear()
                ok = skip = 0
                for i, line in enumerate(text.strip().split("\n")):
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        skip += 1
                        continue
                    try:
                        rec = json.loads(line)
                        norm = _coerce(rec, i)
                        _ingest_buffer.append(norm)
                        ok += 1
                        ingest_log.push(
                            f"✓ row {i+1}: lto={norm['output_tokens']['language_token_output']}"
                        )
                    except Exception as ex:
                        skip += 1
                        ingest_log.push(f"✗ row {i+1}: {ex}")

                ingest_ok_lbl.set_text(f"valid: {ok}")
                ingest_skip_lbl.set_text(f"skipped: {skip}")
                ingest_total_lbl.set_text(f"total: {ok+skip}")

            def _export_ingested():
                if not _ingest_buffer:
                    ingest_log.push("Nothing ingested yet.")
                    return
                ts = int(time.time())
                out = OUTPUT_DIR / f"corpus_ingested_{ts}.jsonl"
                out.write_text("\n".join(json.dumps(r) for r in _ingest_buffer))
                ingest_log.push(f"✅ Exported {len(_ingest_buffer)} records → {out.name}")

            ui.button("Export Ingested JSONL", on_click=_export_ingested).props(
                "dense outline color=orange"
            )

        # ══════════════════════════════════════════════════════════════════════
        # PANEL E — Neural Model Training (PyTorch)
        # ══════════════════════════════════════════════════════════════════════
        with ui.card().classes("w-full border border-purple-300 p-3"):
            ui.label("🧠 Neural Model Training (PyTorch)").classes("font-semibold text-purple-700 text-sm")
            ui.label(
                "Asynchronously train the V31Model over an ingested JSONL corpus. Saves to models/v31_neural_model.pt."
            ).classes("text-xs text-slate-500 mb-2")

            # UI Controls
            with ui.row().classes("gap-3 items-end flex-wrap"):
                corpus_options = ["corpus_v31_sample.jsonl", "corpus_v31_ingested.jsonl"]
                try:
                    for p in ROOT_DIR.glob("*.jsonl"):
                        if p.name not in corpus_options:
                            corpus_options.append(p.name)
                    for p in OUTPUT_DIR.glob("*.jsonl"):
                        rel_path = f"output/{p.name}"
                        if rel_path not in corpus_options:
                            corpus_options.append(rel_path)
                except Exception:
                    pass
                
                corpus_path_sel = ui.select(label="Corpus Path", options=corpus_options, value="corpus_v31_sample.jsonl", with_input=True).classes("w-64")
                epochs_input = ui.number(label="Epochs", value=10, min=1, max=200, step=1).classes("w-20")
                batch_input = ui.number(label="Batch Size", value=16, min=1, max=128, step=1).classes("w-20")
                lr_input = ui.number(label="Learning Rate", value=0.001, min=1e-5, max=0.1, format="%.5f").classes("w-24")
                train_btn = ui.button("Start Training").props("dense color=purple icon=school")

            # Live loss reduction chart
            ui.label("Epoch Loss Reduction Chart:").classes("text-xs text-slate-500 mt-2")
            train_chart = ui.echart({
                "title": {"text": "Training Loss", "textStyle": {"fontSize": 11}},
                "tooltip": {"trigger": "axis"},
                "legend": {"data": ["Total Loss", "MSE (Visual)", "CE (Action)"], "textStyle": {"fontSize": 9}, "top": 18},
                "grid": {"top": 45, "bottom": 24, "left": 45, "right": 15},
                "xAxis": {"type": "category", "data": [], "axisLabel": {"fontSize": 8}},
                "yAxis": {"type": "value", "axisLabel": {"fontSize": 9}},
                "series": [
                    {"name": "Total Loss", "type": "line", "data": [], "itemStyle": {"color": "#a855f7"}},
                    {"name": "MSE (Visual)", "type": "line", "data": [], "itemStyle": {"color": "#3b82f6"}},
                    {"name": "CE (Action)", "type": "line", "data": [], "itemStyle": {"color": "#ec4899"}},
                ],
            }).classes("w-full h-44")

            # Training log
            train_log = ui.log(max_lines=30).classes(
                "w-full text-xs font-mono h-28 bg-slate-900 text-purple-300 mt-1"
            )

            # Training process
            async def _run_training():
                train_btn.disable()
                train_btn.set_text("Training...")
                train_log.clear()
                
                corpus_path = str(corpus_path_sel.value)
                full_path = ROOT_DIR / corpus_path
                if not full_path.exists():
                    train_log.push(f"❌ Error: File not found at {full_path}")
                    train_btn.enable()
                    train_btn.set_text("Start Training")
                    return

                epochs = int(epochs_input.value or 10)
                batch_size = int(batch_input.value or 16)
                lr = float(lr_input.value or 0.001)

                train_log.push(f"▶ Loading corpus from {corpus_path}...")
                train_log.push(f"  Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
                
                try:
                    model = V31Model(vocab_size=5000, num_experts=4)
                    trainer = AsymmetricTrainer(model, lr=lr)
                    
                    records = trainer.load_corpus_data(str(full_path))
                    if not records:
                        train_log.push(f"❌ Error: No valid v31 records loaded from {corpus_path}!")
                        train_btn.enable()
                        train_btn.set_text("Start Training")
                        return
                    
                    train_log.push(f"✓ Loaded {len(records)} training frames.")
                    train_log.push("▶ Initializing optimization loops...")
                    await asyncio.sleep(0.1)

                    epochs_list = []
                    total_loss_list = []
                    mse_loss_list = []
                    ce_loss_list = []

                    train_chart.options["xAxis"]["data"] = epochs_list
                    train_chart.options["series"][0]["data"] = total_loss_list
                    train_chart.options["series"][1]["data"] = mse_loss_list
                    train_chart.options["series"][2]["data"] = ce_loss_list
                    train_chart.update()

                    for epoch in range(1, epochs + 1):
                        model.train()
                        epoch_loss = 0.0
                        epoch_mse = 0.0
                        epoch_ce = 0.0
                        num_batches = 0

                        for i in range(0, len(records), batch_size):
                            batch = records[i : i + batch_size]
                            
                            m1_b = torch.tensor([f["m1"] for f in batch], dtype=torch.float32)
                            m2_b = torch.tensor([f["m2"] for f in batch], dtype=torch.float32)
                            m3_b = torch.tensor([f["m3"] for f in batch], dtype=torch.float32)
                            geo_b = torch.stack([f["geo"] for f in batch])
                            bin_b = torch.stack([f["bin"] for f in batch])
                            lng_b = torch.stack([f["lng"] for f in batch])
                            tri_b = torch.stack([f["tri"] for f in batch])
                            
                            target_frame_b = torch.stack([f["next_frame_target"] for f in batch])
                            target_token_b = torch.tensor([f["next_token_id"] for f in batch], dtype=torch.long)

                            trainer.optimizer.zero_grad()
                            pred_frame, pred_token_logits = model(m1_b, m2_b, m3_b, geo_b, bin_b, lng_b, tri_b)
                            
                            loss_mse = trainer.criterion_mse(pred_frame, target_frame_b)
                            loss_ce = trainer.criterion_ce(pred_token_logits, target_token_b)
                            loss = loss_mse + 0.1 * loss_ce
                            
                            loss.backward()
                            trainer.optimizer.step()
                            
                            epoch_loss += loss.item()
                            epoch_mse += loss_mse.item()
                            epoch_ce += loss_ce.item()
                            num_batches += 1
                            
                            await asyncio.sleep(0.001)

                        avg_loss = epoch_loss / max(num_batches, 1)
                        avg_mse = epoch_mse / max(num_batches, 1)
                        avg_ce = epoch_ce / max(num_batches, 1)

                        train_log.push(
                            f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.5f} | "
                            f"MSE: {avg_mse:.5f} | CE: {avg_ce:.5f}"
                        )

                        epochs_list.append(f"E{epoch}")
                        total_loss_list.append(avg_loss)
                        mse_loss_list.append(avg_mse)
                        ce_loss_list.append(avg_ce)

                        train_chart.options["xAxis"]["data"] = epochs_list
                        train_chart.options["series"][0]["data"] = total_loss_list
                        train_chart.options["series"][1]["data"] = mse_loss_list
                        train_chart.options["series"][2]["data"] = ce_loss_list
                        train_chart.update()

                        await asyncio.sleep(0.05)

                    models_dir = ROOT_DIR / "models"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = models_dir / "v31_neural_model.pt"
                    trainer.save_checkpoint(str(ckpt_path))
                    
                    train_log.push("────────────────────────────────────────────────────────────────")
                    train_log.push(f"✅ Training COMPLETED successfully!")
                    train_log.push(f"💾 Checkpoint saved → models/v31_neural_model.pt")
                    ui.notify("Neural model training complete!", type="positive")
                except Exception as ex:
                    train_log.push(f"❌ Training error: {ex}")
                    ui.notify(f"Training failed: {ex}", type="negative")
                finally:
                    train_btn.enable()
                    train_btn.set_text("Start Training")

            train_btn.on("click", _run_training)
