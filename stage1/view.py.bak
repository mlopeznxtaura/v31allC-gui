from nicegui import ui
from stage1.core.m_scalars import M1State, M2State, M3State, compute_m_scalars
from stage1.core.triangulation import TriangulationState, triangulate
from stage1.core.record_builder import RecordBuilder
from gui.stage1.ingest import Stage1Ingest
from gui.shared.smart_recorder import SmartRecorder, EventType
import numpy as np
import json
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO

_rb = RecordBuilder(total_entries=50, codebase_hash="v31all", mission="gui_run")
_recorder = SmartRecorder(session_name="stage1_gui_session")
_m1 = M1State()
_m2 = M2State()
_m3 = M3State()
_tri = TriangulationState()
_records = []
_v_history = []

def render_pixel_frame(frame_array):
    """Convert numpy array to base64 PNG"""
    try:
        img = Image.fromarray(frame_array.astype('uint8'), 'RGB')
        img = img.resize((256, 256), Image.NEAREST)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Pixel render error: {e}")
        return None

def compute_and_display(stimulus, m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, record_log, binary_input, geometry_input, language_input, pixel_display):
    vals = compute_m_scalars(_m1, _m2, _m3, stimulus=stimulus)
    
    m1_val_label.set_text(f"{vals['M1']:.4f}")
    m2_val_label.set_text(f"{vals['M2']:.4f}")
    m3_val_label.set_text(f"{vals['M3']:.4f}")
    
    m1_branch_label.set_text(f"Branches: {_m1.branch_count}")
    m2_budget_label.set_text(f"Budget: {_m2.cost_budget:.3f}")
    m3_mult_label.set_text(f"Multiplier: {_m3.logic_multiplier:.2f}")
    
    # Auto-populate all 7 inputs from computed state
    binary_input.set_value(f"stimulus={stimulus:.2f} branches={_m1.branch_count}")
    geometry_input.set_value(f"M1={vals['M1']:.4f} M2={vals['M2']:.4f} M3={vals['M3']:.4f}")
    language_input.set_value(f"budget={_m2.cost_budget:.3f} multiplier={_m3.logic_multiplier:.2f} depth={_m1.exploration_depth:.2f}")
    
    # Build record with all 7 inputs
    rec = _rb.build(
        binary_text=binary_input.value,
        geometry_text=geometry_input.value,
        language_text=language_input.value,
    )
    
    # Accumulate records for export
    _records.append(rec)
    
    v = rec["_meta"]["V"]
    nfp = rec["output_tokens"]["next_frame_prediction"]
    lto = rec["output_tokens"]["language_token_output"]
    
    v_label.set_text(f"{v:.4f}")
    _v_history.append(v)
    if len(_v_history) > 5:
        _v_history.pop(0)
    avg_momentum = sum(_v_history) / len(_v_history)
    momentum_label.set_text(f"{avg_momentum:.4f}")
    
    nfp_label.set_text(str(nfp)[:100])
    lto_label.set_text(str(lto)[:120])
    
    # Render pixel frame
    if "pixel_frame_base64" in rec["_meta"]:
        try:
            frame_bytes = base64.b64decode(rec["_meta"]["pixel_frame_base64"])
            frame_shape = rec["_meta"]["pixel_frame_shape"]
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)
            img_base64 = render_pixel_frame(frame_array)
            if img_base64:
                pixel_display.set_source(f"data:image/png;base64,{img_base64}")
        except Exception as e:
            record_log.push(f"Pixel error: {e}")
    
    _recorder.log_scalar_tick(
        M1=vals['M1'],
        M2=vals['M2'],
        M3=vals['M3'],
        V=v,
        stimulus=stimulus
    )
    
    record_log.push(f"✓ M1={vals['M1']:.4f} M2={vals['M2']:.4f} M3={vals['M3']:.4f} V={v:.4f}")
    record_log.push(f"  NFP: {nfp}")
    record_log.push(f"  LTO: {lto}")

def handle_upload(e, ingest_log, ingest_stats):
    try:
        content = e.content.read().decode()
        fmt = Stage1Ingest.detect_format(content)
        parsed = Stage1Ingest.parse_format(content, fmt)
        
        _records.extend(parsed)
        ingest_log.push(f"✓ Uploaded ({fmt}) → {len(parsed)} records")
        ingest_stats.set_text(f"Ingested {len(_records)} total records")
        
        _recorder.log_event(
            EventType.USER_INPUT,
            message=f"File uploaded: {e.name}",
            data={"filename": e.name, "format": fmt, "records": len(parsed)}
        )
    except Exception as ex:
        ingest_log.push(f"✗ Error: {str(ex)}")
        _recorder.log_error(f"Upload failed for {e.name}", ex)

def handle_paste(content, ingest_log, ingest_stats):
    try:
        fmt = Stage1Ingest.detect_format(content)
        parsed = Stage1Ingest.parse_format(content, fmt)
        
        _records.extend(parsed)
        ingest_log.push(f"✓ Pasted ({fmt}) → {len(parsed)} records")
        ingest_stats.set_text(f"Ingested {len(_records)} total records")
        
        _recorder.log_event(
            EventType.USER_INPUT,
            message=f"Data pasted: {fmt} format",
            data={"format": fmt, "records": len(parsed)}
        )
        
        # Auto-compute scalars after ingest
        _auto_compute_after_ingest()
    except Exception as ex:
        ingest_log.push(f"✗ Error: {str(ex)}")
        _recorder.log_error(f"Paste failed", ex)

def build_record(binary_input, geometry_input, language_input, record_log, pixel_display):
    try:
        rec = _rb.build(
            binary_text=binary_input.value or "default",
            geometry_text=geometry_input.value or "default",
            language_text=language_input.value or "default",
        )
        
        # Accumulate records for export
        _records.append(rec)
        
        nfp = rec["output_tokens"]["next_frame_prediction"]
        lto = rec["output_tokens"]["language_token_output"]
        
        record_log.push(f"✓ Record {_rb._entry} built")
        record_log.push(f"  NFP: {nfp}")
        record_log.push(f"  LTO: {lto}")
        
        # Render frame
        if "pixel_frame_base64" in rec["_meta"]:
            frame_bytes = base64.b64decode(rec["_meta"]["pixel_frame_base64"])
            frame_shape = rec["_meta"]["pixel_frame_shape"]
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)
            img_base64 = render_pixel_frame(frame_array)
            if img_base64:
                pixel_display.set_source(f"data:image/png;base64,{img_base64}")
        
        _recorder.log_record_built(
            entry_num=_rb._entry,
            binary_text=binary_input.value,
            geometry_text=geometry_input.value,
            language_text=language_input.value,
            output_tokens=rec["output_tokens"]
        )
    except Exception as ex:
        record_log.push(f"✗ Error: {str(ex)}")
        _recorder.log_error("Record build failed", ex)

def export_jsonl():
    try:
        # Export both ingested records and built records
        all_records = []
        
        # Add ingested records
        all_records.extend(_records)
        
        # Serialize to JSONL
        jsonl_lines = []
        for rec in all_records:
            if isinstance(rec, dict):
                jsonl_lines.append(json.dumps(rec))
        
        if not jsonl_lines:
            print("⚠ No records to export")
            _recorder.log_event(EventType.ERROR, "Export: no records found")
            return
        
        timestamp = int(__import__('time').time())
        out_path = Path(f"/mnt/d/NextAura/v31all_1/v31allC/output/stage1_records_export_{timestamp}.jsonl")
        out_path.write_text('\n'.join(jsonl_lines))
        
        print(f"✅ Exported {len(jsonl_lines)} records to {out_path}")
        _recorder.log_button_click("Export JSONL", f"Exported {len(jsonl_lines)} records to {out_path.name}")
    except Exception as ex:
        print(f"❌ Export error: {ex}")
        _recorder.log_error("Export failed", ex)

def save_session_log():
    out_path = _recorder.save_session("stage1_gui_run")
    _recorder.summary()


def _auto_compute_after_ingest():
    """Auto-trigger computation after data ingest"""
    global _m1, _m2, _m3, _tri
    try:
        stimulus = 1.0
        vals = compute_m_scalars(_m1, _m2, _m3, stimulus=stimulus)
        
        # Build record
        rec = _rb.build(
            binary_text=f"stimulus={stimulus:.2f} auto-computed",
            geometry_text=f"M1={vals['M1']:.4f} M2={vals['M2']:.4f} M3={vals['M3']:.4f}",
            language_text=f"auto-ingested and computed",
        )
        
        _recorder.log_event(
            EventType.SCALAR_COMPUTE,
            message="Auto-computed after ingest",
            data={"M1": round(vals['M1'], 4), "M2": round(vals['M2'], 4), "M3": round(vals['M3'], 4)}
        )
    except Exception as e:
        print(f"Auto-compute error: {e}")


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
            ui.button("Compute M-Scalars", on_click=lambda: compute_and_display(stimulus_input.value, m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, record_log, binary_input, geometry_input, language_input, pixel_display)).props("dense")
        
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
        
        ui.separator()
        
        # === PIXEL FRAME DISPLAY ===
        ui.label("Generated Pixel Frame").classes("text-lg font-bold text-amber-700")
        pixel_display = ui.image().classes("w-full max-w-xs border-2 border-amber-300 rounded")
        pixel_display.set_source("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        
        ui.separator()
        
        # === TRIANGULATION OUTPUT ===
        ui.label("Triangulation (RL Value Function)").classes("text-lg font-bold")
        with ui.row().classes("gap-2 w-full"):
            v_card = ui.card().classes("w-1/3")
            with v_card:
                ui.label("V(s)").classes("text-sm font-semibold")
                v_label = ui.label("—").classes("text-xl font-mono")
            
            momentum_card = ui.card().classes("w-1/3")
            with momentum_card:
                ui.label("Momentum (5-window)").classes("text-sm font-semibold")
                momentum_label = ui.label("—").classes("text-xl font-mono")
        
        ui.separator()
        
        # === OUTPUT TOKENS ===
        ui.label("Output Tokens").classes("text-lg font-bold")
        with ui.row().classes("gap-2 w-full"):
            nfp_card = ui.card().classes("w-1/2")
            with nfp_card:
                ui.label("Next Frame Prediction").classes("text-sm font-semibold text-cyan-700")
                nfp_label = ui.label("—").classes("text-xs font-mono")
            
            lto_card = ui.card().classes("w-1/2")
            with lto_card:
                ui.label("Language Token Output").classes("text-sm font-semibold text-emerald-700")
                lto_label = ui.label("—").classes("text-xs font-mono")
        
        ui.separator()
        
        # === RECORD BUILDER (auto-populated) ===
        ui.label("Record Builder (Auto-Populated)").classes("text-lg font-bold")
        with ui.row().classes("gap-2 w-full"):
            binary_input = ui.textarea(label="Binary Context").classes("flex-1 h-20")
            geometry_input = ui.textarea(label="Geometry Context").classes("flex-1 h-20")
        language_input = ui.textarea(label="Language Context").classes("w-full h-20")
        
        with ui.row().classes("gap-2"):
            ui.button("Build Record", on_click=lambda: build_record(binary_input, geometry_input, language_input, record_log, pixel_display)).props("dense color=cyan")
            ui.button("Export JSONL", on_click=export_jsonl).props("dense color=green")
            ui.button("Save Session Log", on_click=save_session_log).props("dense color=orange")
        
        # === ACTIVITY LOG ===
        ui.label("Activity Log").classes("text-lg font-bold")
        record_log = ui.log(max_lines=30).classes("w-full text-xs font-mono h-40 bg-slate-900")
        
        _recorder.log_event(EventType.GUI_STARTUP, message="Stage1 GUI loaded")
