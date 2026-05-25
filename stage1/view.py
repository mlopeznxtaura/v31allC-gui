from nicegui import ui
from stage1.core.m_scalars import M1State, M2State, M3State, compute_m_scalars
from stage1.core.triangulation import TriangulationState, triangulate
from stage1.core.record_builder import RecordBuilder
from gui.stage1.ingest import Stage1Ingest
import json
from pathlib import Path

_rb = RecordBuilder(total_entries=50, codebase_hash="v31all", mission="gui_run")
_m1 = M1State()
_m2 = M2State()
_m3 = M3State()
_tri = TriangulationState()
_records = []

def stage1_view() -> None:
    with ui.column().classes("w-full gap-4"):
        # === INGEST PANEL ===
        with ui.card().classes("w-full border-2 border-dashed border-cyan-400 p-4"):
            ui.label("🔄 Ingest Raw Data").classes("text-lg font-bold text-cyan-700")
            ui.label("Drop files here or paste JSON/CSV/text weights").classes("text-xs text-slate-600")
            
            with ui.row().classes("w-full gap-2"):
                upload_area = ui.upload(on_upload=lambda e: handle_upload(e, ingest_log, ingest_stats)).classes("flex-1")
                upload_area.props("accept=.json,.csv,.txt,.jsonl max-file-size=52428800")
            
            with ui.row().classes("gap-2 w-full"):
                paste_input = ui.textarea(label="Or paste JSON/CSV/weights", placeholder="Paste content here...").classes("flex-1 h-24")
                ui.button("Ingest Pasted", on_click=lambda: handle_paste(paste_input.value, ingest_log, ingest_stats)).props("dense color=cyan")
            
            ingest_log = ui.log(max_lines=8).classes("w-full text-xs font-mono h-24 bg-slate-900 text-cyan-300")
            ingest_stats = ui.label("Ready").classes("text-xs text-slate-500 mt-2")

        ui.separator()

        # === M-SCALAR CONTROLS ===
        ui.label("M-Scalar Engine").classes("text-lg font-bold")
        with ui.row().classes("gap-2"):
            stimulus_input = ui.number(label="Stimulus", value=1.0, min=0, max=100, step=0.1)
            ui.button("Compute M-Scalars", on_click=lambda: compute_and_display(stimulus_input.value, m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, record_log)).props("dense")

        # === M-SCALAR VALUES ===
        with ui.row().classes("gap-2 w-full"):
            m1_card = ui.card().classes("w-1/3")
            with m1_card:
                ui.label("M1 Unbounded").classes("text-sm font-semibold text-cyan-700")
                m1_val_label = ui.label("—").classes("text-2xl font-mono")
                m1_branch_label = ui.label("Branches: —").classes("text-xs text-slate-600")
            
            m2_card = ui.card().classes("w-1/3")
            with m2_card:
                ui.label("M2 Efficiency").classes("text-sm font-semibold text-emerald-700")
                m2_val_label = ui.label("—").classes("text-2xl font-mono")
                m2_budget_label = ui.label("Budget: —").classes("text-xs text-slate-600")
            
            m3_card = ui.card().classes("w-1/3")
            with m3_card:
                ui.label("M3 Meta").classes("text-sm font-semibold text-rose-700")
                m3_val_label = ui.label("—").classes("text-2xl font-mono")
                m3_mult_label = ui.label("Multiplier: —").classes("text-xs text-slate-600")

        # === TRIANGULATION ===
        ui.label("Triangulation (RL Value Function)").classes("text-lg font-bold mt-4")
        with ui.row().classes("gap-2 w-full"):
            v_card = ui.card().classes("flex-1")
            with v_card:
                ui.label("V(s)").classes("text-sm font-semibold")
                v_label = ui.label("—").classes("text-3xl font-mono font-bold text-cyan-600")
            
            momentum_card = ui.card().classes("flex-1")
            with momentum_card:
                ui.label("Momentum (5-window)").classes("text-sm font-semibold")
                momentum_label = ui.label("—").classes("text-3xl font-mono font-bold text-slate-600")

        # === OUTPUT TOKENS ===
        ui.label("Output Tokens").classes("text-lg font-bold mt-4")
        with ui.row().classes("gap-2 w-full"):
            nfp_card = ui.card().classes("flex-1")
            with nfp_card:
                ui.label("Next Frame Prediction").classes("text-xs font-semibold text-slate-500")
                nfp_label = ui.label("—").classes("text-sm font-mono italic")
            
            lto_card = ui.card().classes("flex-1")
            with lto_card:
                ui.label("Language Token Output").classes("text-xs font-semibold text-slate-500")
                lto_label = ui.label("—").classes("text-sm font-mono font-bold text-cyan-700")

        # === RECORD BUILDER ===
        ui.separator()
        ui.label("Record Builder").classes("text-lg font-bold")
        with ui.column().classes("gap-2 w-full"):
            binary_input = ui.textarea(label="Binary Context", value="Logical init phase").classes("w-full")
            geometry_input = ui.textarea(label="Geometry Context", value="Entry in space-time").classes("w-full")
            language_input = ui.textarea(label="Language Context", value="Rich semantic narrative").classes("w-full")
            
            def build_record():
                rec = _rb.build(
                    binary_text=binary_input.value,
                    geometry_text=geometry_input.value,
                    language_text=language_input.value,
                )
                _records.append(rec)
                record_log.push(f"✓ Record {len(_records)} built")
                v_label.set_text(f"{rec['_meta']['V']:.4f}")
                m1_val_label.set_text(f"{rec['_meta']['M1_raw']:.4f}")
                m2_val_label.set_text(f"{rec['_meta']['M2_raw']:.4f}")
                m3_val_label.set_text(f"{rec['_meta']['M3_raw']:.4f}")
                nfp_label.set_text(rec["output_tokens"]["next_frame_prediction"])
                lto_label.set_text(rec["output_tokens"]["language_token_output"])
            
            with ui.row().classes("gap-2"):
                ui.button("Build Record", on_click=build_record).props("dense color=cyan")
                ui.button("Export JSONL", on_click=lambda: export_jsonl()).props("dense color=green")

        # === RECORD LOG ===
        ui.label("Activity Log").classes("text-lg font-bold mt-4")
        record_log = ui.log(max_lines=15).classes("w-full text-xs font-mono h-32")

def handle_upload(e, log, stats):
    """Handle file upload."""
    log.clear()
    if e.content:
        content = e.content.read().decode()
        parsed, fmt = Stage1Ingest.ingest_file(content, e.name)
        log.push(f"✓ {e.name} ({fmt}) → {len(parsed)} records")
        stats.set_text(f"Ingested {len(parsed)} records from {e.name}")
        jsonl = Stage1Ingest.to_stage1_jsonl(parsed, e.name)
        out_path = Path(f"/mnt/d/NextAura/v31all_1/v31allC/output/stage1_ingest_{e.name}.jsonl")
        out_path.write_text(jsonl)
        log.push(f"✓ Saved to {out_path}")

def handle_paste(content, log, stats):
    """Handle pasted content."""
    log.clear()
    if not content.strip():
        log.push("✗ Empty content")
        return
    parsed, fmt = Stage1Ingest.ingest_file(content, "pasted")
    log.push(f"✓ Pasted ({fmt}) → {len(parsed)} records")
    stats.set_text(f"Ingested {len(parsed)} records from paste")
    jsonl = Stage1Ingest.to_stage1_jsonl(parsed, "pasted")
    out_path = Path(f"/mnt/d/NextAura/v31all_1/v31allC/output/stage1_ingest_pasted.jsonl")
    out_path.write_text(jsonl)
    log.push(f"✓ Saved to {out_path}")

def export_jsonl():
    """Export built records."""
    if not _records:
        return
    jsonl = '\n'.join(json.dumps(r) for r in _records)
    out_path = Path(f"/mnt/d/NextAura/v31all_1/v31allC/output/stage1_records_export.jsonl")
    out_path.write_text(jsonl)
    ui.notify(f"Exported {len(_records)} records to {out_path}")

def compute_and_display(stimulus, m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, record_log):
    vals = compute_m_scalars(_m1, _m2, _m3, stimulus=stimulus)
    v, nfp, lto = triangulate(
        M1=vals["M1"], M2=vals["M2"], M3=vals["M3"],
        binary=0.8, geometry=0.5, language=0.6,
        state=_tri
    )
    m1_val_label.set_text(f"{vals['M1']:.4f}")
    m2_val_label.set_text(f"{vals['M2']:.4f}")
    m3_val_label.set_text(f"{vals['M3']:.4f}")
    m1_branch_label.set_text(f"Branches: {_m1.branch_count}")
    m2_budget_label.set_text(f"Budget: {_m2.cost_budget:.3f}")
    m3_mult_label.set_text(f"Multiplier: {_m3.logic_multiplier:.2f}")
    v_label.set_text(f"{v:.4f}")
    momentum_label.set_text(f"{_tri.momentum():.4f}")
    nfp_label.set_text(nfp)
    lto_label.set_text(lto)
    record_log.push(f"M1={vals['M1']:.3f} M2={vals['M2']:.3f} M3={vals['M3']:.3f} V={v:.4f}")

