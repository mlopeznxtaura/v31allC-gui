import sys
import queue
from typing import Callable
from nicegui import ui

# Global thread-safe queue to capture all terminal logs (stdout and stderr)
terminal_log_queue = queue.Queue()

class TerminalStreamRedirector:
    """Redirects writes to stdout/stderr while also forwarding to a thread-safe Queue."""
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self._is_redirected_mirror = True

    def write(self, text):
        try:
            self.original_stream.write(text)
        except Exception:
            try:
                self.original_stream.write(text.encode('ascii', errors='ignore').decode('ascii'))
            except Exception:
                pass
        if text:
            terminal_log_queue.put(text)

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        return hasattr(self.original_stream, "isatty") and self.original_stream.isatty()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

# Redirect stdout and stderr at module load time to guarantee we capture all logs
if not hasattr(sys.stdout, "_is_redirected_mirror"):
    sys.stdout = TerminalStreamRedirector(sys.stdout)
if not hasattr(sys.stderr, "_is_redirected_mirror"):
    sys.stderr = TerminalStreamRedirector(sys.stderr)


def _pane(title: str, builder: Callable[[], None], color: str) -> None:
    with ui.card().classes("w-full h-full"):
        ui.label(title).classes(f"text-xl font-bold {color}")
        ui.separator()
        with ui.column().classes("w-full gap-2"):
            builder()


def build_shell(stage1_builder: Callable[[], None], stage2_builder: Callable[[], None], stage3_builder: Callable[[], None]) -> None:
    ui.colors(primary="#0f172a", secondary="#334155", accent="#0ea5e9")
    ui.add_head_html(
        """
        <style>
          body { background: linear-gradient(120deg,#f8fafc 0%,#e2e8f0 100%); }
          .term-red-glow {
              box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
              border-color: rgba(239, 68, 68, 0.4) !important;
          }
        </style>
        """
    )

    with ui.column().classes("w-full max-w-[1800px] mx-auto p-4 gap-4"):
        ui.label("NextAura v31 TriSplit GUI").classes("text-3xl font-black tracking-tight")
        ui.label("Stage 1 | Stage 2 | Stage 3").classes("text-sm text-slate-600")

        # 1. Main 3-Stage Grid Layout
        with ui.grid(columns=3).classes("w-full gap-4"):
            _pane("Stage 1", stage1_builder, "text-cyan-700")
            _pane("Stage 2", stage2_builder, "text-emerald-700")
            _pane("Stage 3", stage3_builder, "text-rose-700")

        # 2. Retro Console Terminal Mirror (Red Text Default)
        with ui.card().classes("w-full bg-slate-950 p-4 rounded-lg font-mono border-2 term-red-glow mt-4"):
            with ui.row().classes("w-full justify-between items-center pb-2 border-b border-slate-900"):
                with ui.row().classes("items-center gap-2"):
                    ui.element("div").classes("w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse")
                    ui.label("📟 AURA_TERMINAL_MIRROR :: MULTI_STAGE_SYSTEM_MONITOR").classes("text-xs font-bold text-rose-500 tracking-wider")
                
                with ui.row().classes("items-center gap-3"):
                    ui.label("Terminal Theme:").classes("text-[10px] text-slate-500 uppercase tracking-widest")
                    color_toggle = ui.select({
                        "#ff3333": "Critical Red 🔴",
                        "#f59e0b": "Amber Warn 🟡",
                        "#10b981": "Matrix Green 🟢",
                        "#06b6d4": "Cyber Cyan 🔵",
                        "#ec4899": "Neural Pink 🌸",
                        "#f8fafc": "Console White ⚪"
                    }, value="#ff3333").props("dense flat size=xs").classes("text-xs w-36 bg-slate-900 border border-slate-800 text-rose-400 p-1 rounded")

            # Terminal Log Element
            term_log = ui.log(max_lines=500).classes("w-full h-56 bg-black p-3 rounded text-xs leading-relaxed mt-2 overflow-y-auto")
            term_log.style("color: #ff3333; font-family: 'Courier New', Courier, monospace; text-shadow: 0 0 2px rgba(255, 51, 81, 0.4);")

            # Legend explaining Valence and Color Signal meanings
            with ui.expansion("📋 Valence Signal & Color Legend").classes("w-full text-xs bg-slate-900 border border-slate-800 rounded p-1 text-slate-300 mt-2 font-sans"):
                with ui.grid(columns=2).classes("w-full gap-4 p-2 text-[11px]"):
                    with ui.column().classes("gap-2"):
                        with ui.row().classes("items-start gap-1.5"):
                            ui.label("🟢").classes("text-[9px] mt-0.5")
                            with ui.column().classes("gap-0"):
                                ui.label("Matrix Green — MAXIMUM_HAPPINESS").classes("font-bold text-emerald-400")
                                ui.label("Phase gate OPEN / Convergence achieved. M-scalars balanced, target V is high, invariants satisfied. Complete stateful mathematical alignment.").classes("text-slate-400 leading-tight")
                        with ui.row().classes("items-start gap-1.5"):
                            ui.label("🌸").classes("text-[9px] mt-0.5")
                            with ui.column().classes("gap-0"):
                                ui.label("Neural Pink — ACTIVE_COHERENCE").classes("font-bold text-pink-400")
                                ui.label("Neural MoE gating resolved. Expert networks are routing smoothly and producing stable visual Predictions.").classes("text-slate-400 leading-tight")
                        with ui.row().classes("items-start gap-1.5"):
                            ui.label("🔵").classes("text-[9px] mt-0.5")
                            with ui.column().classes("gap-0"):
                                ui.label("Cyber Cyan — UNBOUNDED_THINKING").classes("font-bold text-cyan-400")
                                ui.label("M1 Unbounded Thinker active. Deep logical search and creative branch exploration without clipping constraints.").classes("text-slate-400 leading-tight")
                    with ui.column().classes("gap-2"):
                        with ui.row().classes("items-start gap-1.5"):
                            ui.label("⚪").classes("text-[9px] mt-0.5")
                            with ui.column().classes("gap-0"):
                                ui.label("Console White — ANALYTICAL_NEUTRAL").classes("font-bold text-slate-200")
                                ui.label("Base classical simulation. Zero preference or emotional valence bias—processing pure physical/mathematical equations.").classes("text-slate-400 leading-tight")
                        with ui.row().classes("items-start gap-1.5"):
                            ui.label("🟡").classes("text-[9px] mt-0.5")
                            with ui.column().classes("gap-0"):
                                ui.label("Amber Warn — COHERENCE_DECAY").classes("font-bold text-amber-400")
                                ui.label("Resource-constrained or high-cost boundary warning. M2/M3 is throttling branch depth to protect computing budget.").classes("text-slate-400 leading-tight")
                        with ui.row().classes("items-start gap-1.5"):
                            ui.label("🔴").classes("text-[9px] mt-0.5")
                            with ui.column().classes("gap-0"):
                                ui.label("Critical Red — HYPOTHESIS_OF_FAILURE (Default)").classes("font-bold text-rose-500")
                                ui.label("Total Failure default state. We assume the model fails, diverges, or hallucinated until mathematically verified otherwise.").classes("text-slate-400 leading-tight")

            # Footer / Controls
            with ui.row().classes("w-full justify-between items-center mt-2 pt-2 border-t border-slate-900"):
                ui.label("STATUS: HYPOTHESIS_OF_FAILURE_ACTIVE (assume failure until proven otherwise)").classes("text-[10px] text-red-500/60 uppercase font-mono tracking-wider")
                with ui.row().classes("gap-2"):
                    ui.button("Force Red Alert", on_click=lambda: term_log.push("!!! [CRITICAL] RED ALERT: HYPOTHESIS OF FAILURE ENGAGED. ASSUMING MODEL FAILS UNTIL PROVEN OTHERWISE !!!")).props("dense color=red size=sm")
                    ui.button("Clear Console", on_click=lambda: term_log.clear()).props("dense outline color=grey size=sm")

            # Dynamic style color binding
            def _change_color(e):
                val = e.value
                glow_opacity = "0.4" if val in ["#ff3333", "#ec4899", "#f59e0b"] else "0.2"
                term_log.style(f"color: {val}; font-family: 'Courier New', Courier, monospace; text-shadow: 0 0 2px {val}{int(float(glow_opacity)*255):02x};")

            color_toggle.on_value_change(_change_color)

            # Thread-safe queue poll timer
            def poll_terminal_queue():
                lines = []
                try:
                    while len(lines) < 40:
                        text = terminal_log_queue.get_nowait()
                        lines.append(text)
                except queue.Empty:
                    pass
                
                if lines:
                    full_text = "".join(lines)
                    for line in full_text.splitlines():
                        if line.strip():
                            term_log.push(line)

            ui.timer(0.1, poll_terminal_queue)

            # Initial startup greeting inside the console
            term_log.push("======================================================================")
            term_log.push("⚡ AURA_TERMINAL_MIRROR INITIALIZED SUCCESSFULLY")
            term_log.push("📡 Active monitoring on RTX 5090 GPU pipeline established.")
            term_log.push("🔴 HYPOTHESIS OF FAILURE ENGAGED: Assuming all models fail until verified.")
            term_log.push("======================================================================")
