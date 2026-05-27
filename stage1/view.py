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

from stage1.core.neural_l1_engine import NeuralL1State
from gui.stage3.view import scan_checkpoints

_neural_l1 = NeuralL1State()
_rb = RecordBuilder(total_entries=50, codebase_hash="v31all", mission="gui_run")
_recorder = SmartRecorder(session_name="stage1_gui_session")
_records = []
_v_history = []

def auto_populate_next_inputs(binary_input, geometry_input, language_input):
    """Auto-populate the inputs for the next entry to be built by _rb"""
    from stage1.core.layer2_scalars import BinaryScalar, GeometryScalar
    b = BinaryScalar(
        codebase_hash=_rb.codebase_hash,
        mission=_rb.mission,
        purpose="",
        is_logical=True,
    )
    g = GeometryScalar(
        entry_index=_rb._entry + 1,
        total_entries=_rb.total_entries,
        sandbox=_rb.sandbox,
        system_id=_rb.system_id,
        telemetry_stable=True,
    )
    binary_input.set_value(b.to_field())
    geometry_input.set_value(g.to_field())
    language_input.set_value(f"State progression cycle for entry {_rb._entry + 1}")

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
    # Auto-populate all 7 inputs from the current (pre-step) state of _rb
    cur_m1 = _rb._m1.exploration_depth / max(_rb._m1.branch_count, 1) if _rb._m1.branch_count else 0.0
    cur_m2 = _rb._m2.cost_budget / (1.0 + _rb._m1.branch_count / max(_rb._m2.steps_allowed, 1))
    cur_m3 = (_rb._m3.logic_multiplier + _rb._m3.efficiency_multiplier + _rb._m3.creative_multiplier) / 3.0
    
    binary_input.set_value(f"stimulus={stimulus:.2f} branches={_rb._m1.branch_count}")
    geometry_input.set_value(f"M1={cur_m1:.4f} M2={cur_m2:.4f} M3={cur_m3:.4f}")
    language_input.set_value(f"budget={_rb._m2.cost_budget:.3f} multiplier={_rb._m3.logic_multiplier:.2f} depth={_rb._m1.exploration_depth:.2f}")
    
    # Build record with all 7 inputs
    rec = _rb.build(
        binary_text=binary_input.value,
        geometry_text=geometry_input.value,
        language_text=language_input.value,
        stimulus=stimulus,
    )
    
    # Accumulate records for export
    _records.append(rec)
    
    # Update UI labels, outputs, and inputs from the built record
    _update_ui_from_last_record(
        m1_val_label, m2_val_label, m3_val_label,
        m1_branch_label, m2_budget_label, m3_mult_label,
        v_label, momentum_label, nfp_label, lto_label,
        binary_input, geometry_input, language_input,
        pixel_display, record_log
    )
    
    v = rec["_meta"]["V"]
    nfp = rec["output_tokens"]["next_frame_prediction"]
    lto = rec["output_tokens"]["language_token_output"]
    m1_val = rec["_meta"]["M1_raw"]
    m2_val = rec["_meta"]["M2_raw"]
    m3_val = rec["_meta"]["M3_raw"]
    
    _recorder.log_scalar_tick(
        M1=m1_val,
        M2=m2_val,
        M3=m3_val,
        V=v,
        stimulus=stimulus
    )
    
    record_log.push(f"✓ M1={m1_val:.4f} M2={m2_val:.4f} M3={m3_val:.4f} V={v:.4f}")
    record_log.push(f"  NFP: {nfp}")
    record_log.push(f"  LTO: {lto}")

def _run_batch(parsed, source_name, ingest_log, ingest_stats):
    """Route parsed records through RecordBuilder.build_batch (GPU language scalars)."""
    rows = []
    for rec in parsed:
        si = rec.get("scalar_inputs", {}) if isinstance(rec.get("scalar_inputs"), dict) else {}
        
        # Support both modern lowcase/nested schema and raw capitalized keys from baby models
        lang = str(si.get("language") or 
                   rec.get("LanguageContext") or 
                   rec.get("language_text") or 
                   rec.get("description") or 
                   rec.get("language") or 
                   rec.get("text") or "")
                
        geo  = str(si.get("geometry") or 
                   rec.get("Geometry") or 
                   rec.get("geometry_text") or 
                   rec.get("analysis") or 
                   rec.get("geometry") or 
                   rec.get("phase") or "")
                
        bin_ = str(si.get("binary") or 
                   rec.get("Binary") or 
                   rec.get("binary_text") or 
                   rec.get("step") or rec.get("entry_id") or rec.get("binary") or "")
                
        if not lang and not geo:
            continue
        original_outputs = rec.get("output_tokens") if isinstance(rec.get("output_tokens"), dict) else None
        rows.append({
            "binary_text": bin_[:300],
            "geometry_text": geo[:400],
            "language_text": lang[:600],
            "original_outputs": original_outputs
        })
    if not rows:
        ingest_log.push("No usable records found.")
        return
    built = _rb.build_batch(rows)
    
    # Run through SOC compliance and deduplication engine
    from stage1.core.soc_compliance import IngestionComplianceEngine
    clean_built, metrics = IngestionComplianceEngine.process_records(built, source_name)
    
    _records.extend(clean_built)
    
    # Auto-grow Master Corpus
    try:
        from stage3.master_corpus.manager import MasterCorpusManager
        added_to_master = MasterCorpusManager.add_clean_records(clean_built)
        if added_to_master > 0:
            ingest_log.push(f"📁 Auto-grew Master Corpus: +{added_to_master} new records.")
            from gui.stage3.view import stage3_view
            if hasattr(stage3_view, "refresh_corpus_stats"):
                stage3_view.refresh_corpus_stats()
    except Exception as e_mc:
        print(f"Failed auto-growing master corpus: {e_mc}")

    vs = [r["_meta"]["V"] for r in clean_built]
    v_mean = sum(vs)/len(vs) if vs else 0
    v_max  = max(vs) if vs else 0
    
    ingest_log.push(f"Cleaned {len(clean_built)} unique records (skipped {metrics['duplicates_skipped']} duplicates).")
    ingest_log.push(f"Corpus Signature: {metrics['corpus_sha256_signature'][:16]}...")
    ingest_stats.set_text(f"{len(_records)} total | last batch V_mean={v_mean:.4f} V_max={v_max:.4f}")
    
    _recorder.log_event(EventType.USER_INPUT, message=f"Batch: {source_name}",
                        data={"records": len(clean_built), "skipped_duplicates": metrics["duplicates_skipped"], "V_mean": round(v_mean,4)})
    
    # If the audit panel is available, refresh it
    if hasattr(stage1_view, "refresh_audit"):
        stage1_view.refresh_audit()

def _update_ui_from_last_record(m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, binary_input, geometry_input, language_input, pixel_display, record_log):
    if not _records:
        return
    rec = _records[-1]
    meta = rec.get("_meta", {})
    
    m1_val_label.set_text(f"{meta.get('M1_raw', 0.0):.4f}")
    m2_val_label.set_text(f"{meta.get('M2_raw', 0.0):.4f}")
    m3_val_label.set_text(f"{meta.get('M3_raw', 0.0):.4f}")
    
    m1_branch_label.set_text(f"Branches: {_rb._m1.branch_count}")
    m2_budget_label.set_text(f"Budget: {_rb._m2.cost_budget:.3f}")
    m3_mult_label.set_text(f"Multiplier: {_rb._m3.logic_multiplier:.2f}")
    
    v = meta.get("V", 0.0)
    v_label.set_text(f"{v:.4f}")
    
    _v_history.append(v)
    if len(_v_history) > 5:
        _v_history.pop(0)
    avg_momentum = sum(_v_history) / len(_v_history) if _v_history else 0.0
    momentum_label.set_text(f"{avg_momentum:.4f}")
    
    nfp = rec.get("output_tokens", {}).get("next_frame_prediction", "—")
    lto = rec.get("output_tokens", {}).get("language_token_output", "—")
    
    nfp_label.set_text(str(nfp)[:100])
    lto_label.set_text(str(lto)[:120])
    
    binary_input.set_value(str(rec.get("scalar_inputs", {}).get("binary", "")))
    geometry_input.set_value(str(rec.get("scalar_inputs", {}).get("geometry", "")))
    language_input.set_value(str(rec.get("scalar_inputs", {}).get("language", "")))
    
    if "pixel_frame_base64" in meta:
        try:
            frame_bytes = base64.b64decode(meta["pixel_frame_base64"])
            frame_shape = meta["pixel_frame_shape"]
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)
            img_base64 = render_pixel_frame(frame_array)
            if img_base64:
                pixel_display.set_source(f"data:image/png;base64,{img_base64}")
        except Exception as e:
            record_log.push(f"Pixel error: {e}")

def handle_upload(e, ingest_log, ingest_stats, m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, binary_input, geometry_input, language_input, pixel_display, record_log):
    try:
        content = e.content.read().decode(errors="replace")
        fmt = Stage1Ingest.detect_format(content)
        parsed = Stage1Ingest.parse_format(content, fmt)
        ingest_log.push(f"Parsed {len(parsed)} records ({fmt}) - GPU batch build...")
        _run_batch(parsed, e.name, ingest_log, ingest_stats)
        _update_ui_from_last_record(m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, binary_input, geometry_input, language_input, pixel_display, record_log)
        auto_populate_next_inputs(binary_input, geometry_input, language_input)
    except Exception as ex:
        ingest_log.push(f"Error: {str(ex)}")
        _recorder.log_error(f"Upload failed for {e.name}", ex)

def handle_paste(content, ingest_log, ingest_stats, m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, binary_input, geometry_input, language_input, pixel_display, record_log):
    try:
        fmt = Stage1Ingest.detect_format(content)
        parsed = Stage1Ingest.parse_format(content, fmt)
        ingest_log.push(f"Parsed {len(parsed)} records ({fmt}) - GPU batch build...")
        _run_batch(parsed, "paste", ingest_log, ingest_stats)
        # _auto_compute_after_ingest()
        _update_ui_from_last_record(m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, binary_input, geometry_input, language_input, pixel_display, record_log)
        auto_populate_next_inputs(binary_input, geometry_input, language_input)
    except Exception as ex:
        ingest_log.push(f"Error: {str(ex)}")
        _recorder.log_error("Paste failed", ex)

def one_shot_auto_ingest(ingest_log, ingest_stats, m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, binary_input, geometry_input, language_input, pixel_display, record_log):
    try:
        pt_corpus_dir = Path("D:/NextAura/v31all_1/Pt's/Corpus")
        if not pt_corpus_dir.exists():
            pt_corpus_dir = Path("/mnt/d/NextAura/v31all_1/Pt's/Corpus")
        if not pt_corpus_dir.exists():
            pt_corpus_dir = Path(__file__).resolve().parent.parent.parent.parent / "Pt's" / "Corpus"
            
        if not pt_corpus_dir.exists():
            ingest_log.push(f"Error: Corpus dir {pt_corpus_dir} not found.")
            return
            
        ingest_log.push("Starting 1-Shot Auto-Ingest of all 13 baby model files...")
        
        all_parsed = []
        for i in range(1, 14):
            file_path = pt_corpus_dir / f"{i}.jsonl"
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8", errors="replace")
                parsed = Stage1Ingest.parse_json(content)
                all_parsed.extend(parsed)
                ingest_log.push(f"  - Loaded {len(parsed)} records from {file_path.name}")
                
        if not all_parsed:
            ingest_log.push("Error: No records loaded.")
            return
            
        ingest_log.push(f"Total parsed: {len(all_parsed)} records. Running GPU batch build...")
        _run_batch(all_parsed, "one_shot_auto_ingest", ingest_log, ingest_stats)
        # _auto_compute_after_ingest()
        _update_ui_from_last_record(m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label, binary_input, geometry_input, language_input, pixel_display, record_log)
        auto_populate_next_inputs(binary_input, geometry_input, language_input)
        ingest_log.push("✅ 1-Shot Ingestion and UI pipeline update complete!")
    except Exception as ex:
        ingest_log.push(f"Error: {str(ex)}")

def build_record(binary_input, geometry_input, language_input, record_log, pixel_display, m1_val_label, m2_val_label, m3_val_label, m1_branch_label, m2_budget_label, m3_mult_label, v_label, momentum_label, nfp_label, lto_label):
    try:
        rec = _rb.build(
            binary_text=binary_input.value or "default",
            geometry_text=geometry_input.value or "default",
            language_text=language_input.value or "default",
        )
        
        # Accumulate records for export
        _records.append(rec)
        
        _update_ui_from_last_record(
            m1_val_label, m2_val_label, m3_val_label,
            m1_branch_label, m2_budget_label, m3_mult_label,
            v_label, momentum_label, nfp_label, lto_label,
            binary_input, geometry_input, language_input,
            pixel_display, record_log
        )
        auto_populate_next_inputs(binary_input, geometry_input, language_input)
        
        record_log.push(f"✓ Record {_rb._entry} built")
        record_log.push(f"  NFP: {rec['output_tokens']['next_frame_prediction']}")
        record_log.push(f"  LTO: {rec['output_tokens']['language_token_output']}")
        
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
        root_dir = Path(__file__).resolve().parent.parent.parent
        out_path = root_dir / "output" / f"stage1_records_export_{timestamp}.jsonl"
        out_path.write_text('\n'.join(jsonl_lines))
        
        print(f"✅ Exported {len(jsonl_lines)} records to {out_path}")
        _recorder.log_button_click("Export JSONL", f"Exported {len(jsonl_lines)} records to {out_path.name}")
    except Exception as ex:
        print(f"❌ Export error: {ex}")
        _recorder.log_error("Export failed", ex)

def save_session_log():
    out_path = _recorder.save_session("stage1_gui_run")
    _recorder.summary()


def update_engine_mode(mode_val, checkpoint_val, record_log):
    if mode_val == "Neural":
        if checkpoint_val == "simulated":
            _rb.neural_l1 = None
            _neural_l1.is_loaded = False
            record_log.push("[Engine] L1 mode set to Neural, but checkpoint is 'simulated' (using fallback).")
        else:
            success = _neural_l1.load_weights(checkpoint_val)
            if success:
                _rb.neural_l1 = _neural_l1
                record_log.push(f"[Engine] Loaded Layer-1 neural weights: {Path(checkpoint_val).name}")
            else:
                _rb.neural_l1 = None
                record_log.push(f"[Engine] Warning: Failed to load L1 neural weights from {checkpoint_val}. Falling back.")
    else:
        _rb.neural_l1 = None
        record_log.push("[Engine] Mode set to Analytical (Rule-Based).")


def _auto_compute_after_ingest():
    """Auto-trigger computation after data ingest"""
    try:
        stimulus = 1.0
        if _rb.neural_l1 is not None and _rb.neural_l1.is_loaded:
            progress = _rb._entry / max(_rb.total_entries, 1)
            prev_lang = 0.5
            prev_bin = 0.5
            if _records:
                prev_meta = _records[-1].get("_meta", {})
                prev_lang = prev_meta.get("language_scalar", 0.5)
                prev_bin = prev_meta.get("binary_scalar", 0.5)
                
            m1_val, m2_val, m3_val = _rb.neural_l1.step(
                progress=progress,
                language_scalar=prev_lang,
                binary_scalar=prev_bin,
                stimulus=stimulus
            )
            _ = compute_m_scalars(_rb._m1, _rb._m2, _rb._m3, stimulus=stimulus)
            vals = {"M1": m1_val, "M2": m2_val, "M3": m3_val}
        else:
            vals = compute_m_scalars(_rb._m1, _rb._m2, _rb._m3, stimulus=stimulus)
        
        # Build record
        rec = _rb.build(
            binary_text=f"stimulus={stimulus:.2f} auto-computed",
            geometry_text=f"M1={vals['M1']:.4f} M2={vals['M2']:.4f} M3={vals['M3']:.4f}",
            language_text=f"auto-ingested and computed",
            stimulus=stimulus,
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
                upload_area = ui.upload(on_upload=lambda e: handle_upload(
                    e, ingest_log, ingest_stats,
                    m1_val_label, m2_val_label, m3_val_label,
                    m1_branch_label, m2_budget_label, m3_mult_label,
                    v_label, momentum_label, nfp_label, lto_label,
                    binary_input, geometry_input, language_input,
                    pixel_display, record_log
                )).classes("flex-1")
                upload_area.props("accept=.json,.csv,.txt,.jsonl max-file-size=52428800")
            with ui.row().classes("gap-2 w-full items-center"):
                paste_input = ui.textarea(label="Or paste JSON/CSV/weights", placeholder="Paste content here...").classes("flex-1 h-24")
                with ui.column().classes("gap-2"):
                    ui.button("Ingest Pasted", on_click=lambda: handle_paste(
                        paste_input.value, ingest_log, ingest_stats,
                        m1_val_label, m2_val_label, m3_val_label,
                        m1_branch_label, m2_budget_label, m3_mult_label,
                        v_label, momentum_label, nfp_label, lto_label,
                        binary_input, geometry_input, language_input,
                        pixel_display, record_log
                    )).props("dense color=cyan")
                    ui.button("⚡ 1-SHOT AUTO-INGEST", on_click=lambda: one_shot_auto_ingest(
                        ingest_log, ingest_stats,
                        m1_val_label, m2_val_label, m3_val_label,
                        m1_branch_label, m2_budget_label, m3_mult_label,
                        v_label, momentum_label, nfp_label, lto_label,
                        binary_input, geometry_input, language_input,
                        pixel_display, record_log
                    )).props("dense color=teal").classes("font-bold")
            ingest_log = ui.log(max_lines=8).classes("w-full text-xs font-mono h-24 bg-slate-900 text-cyan-300")
            ingest_stats = ui.label("Ready").classes("text-xs text-slate-500 mt-2")
            
            # --- SOC 2/3 Compliance Ingestion Logs ---
            with ui.expansion("📋 SOC 2/3 Compliance Ingestion Logs", icon="verified_user").classes("w-full border border-cyan-100 rounded mt-2 bg-slate-950"):
                with ui.row().classes("w-full justify-between items-center px-2 py-1"):
                    ui.label("Audited Data Lineage & Signatures").classes("text-[11px] font-semibold text-cyan-300 font-mono")
                    ui.button("Load History", on_click=lambda: _load_audit_history()).props("dense color=cyan text-xs outline")
                audit_log = ui.log(max_lines=15).classes("w-full text-xs font-mono h-24 bg-slate-950 text-cyan-200 p-1")
                
                def _load_audit_history():
                    audit_log.clear()
                    from stage1.core.soc_compliance import IngestionComplianceEngine
                    history = IngestionComplianceEngine.get_audit_history()
                    if not history:
                        audit_log.push("No compliance records found.")
                        return
                    for h in history:
                        audit_log.push(f"[{h['timestamp']}] {h['source']}: Raw={h['raw_record_count']} | Clean={h['unique_record_count']} | Skip={h['duplicates_skipped']}")
                        audit_log.push(f"  └─ Signature: {h['corpus_sha256_signature']}")
                
                # Expose the refresh function for auto-trigger
                stage1_view.refresh_audit = _load_audit_history
                # Run once on startup
                ui.timer(0.5, _load_audit_history, once=True)
        
        ui.separator()
        
        # === M-SCALAR CONTROLS ===
        ui.label("M-Scalar Engine").classes("text-lg font-bold")
        
        # New Model & Engine Configuration Panel (L1 weight selector & mode toggle)
        with ui.row().classes("w-full items-center gap-4 bg-slate-900/50 p-2 rounded border border-cyan-500/20"):
            ui.label("L1 Engine Mode:").classes("text-xs font-semibold text-slate-400 font-mono")
            engine_mode_toggle = ui.radio(["Analytical", "Neural"], value="Analytical").props("inline color=cyan text-color=slate-100")
            
            ui.label("L1 Weight Checkpoint:").classes("text-xs font-semibold text-slate-400 font-mono ml-4")
            
            l1_files, _ = scan_checkpoints()
            l1_options = {f["value"]: f["label"] for f in l1_files}
            
            default_l1 = "simulated"
            for f in l1_files:
                if "layer1_combined" in f["name"]:
                    default_l1 = f["value"]
                    break
            
            l1_sel = ui.select(options=l1_options, value=default_l1).classes("w-72").props("dense outline color=cyan")
            
            # Change listeners
            def on_engine_change():
                update_engine_mode(engine_mode_toggle.value, l1_sel.value, record_log)
                
            engine_mode_toggle.on_value_change(on_engine_change)
            l1_sel.on_value_change(on_engine_change)

        with ui.row().classes("gap-2 mt-2"):
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
        
        auto_populate_next_inputs(binary_input, geometry_input, language_input)
        
        with ui.row().classes("gap-2"):
            ui.button("Build Record", on_click=lambda: build_record(
                binary_input, geometry_input, language_input, record_log, pixel_display,
                m1_val_label, m2_val_label, m3_val_label,
                m1_branch_label, m2_budget_label, m3_mult_label,
                v_label, momentum_label, nfp_label, lto_label
            )).props("dense color=cyan")
            ui.button("Export JSONL", on_click=export_jsonl).props("dense color=green")
            ui.button("Save Session Log", on_click=save_session_log).props("dense color=orange")
        
        # === ACTIVITY LOG ===
        ui.label("Activity Log").classes("text-lg font-bold")
        record_log = ui.log(max_lines=30).classes("w-full text-xs font-mono h-40 bg-slate-900")
        
        _recorder.log_event(EventType.GUI_STARTUP, message="Stage1 GUI loaded")
