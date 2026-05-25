from typing import Callable
from nicegui import ui


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
        </style>
        """
    )

    with ui.column().classes("w-full max-w-[1800px] mx-auto p-4 gap-4"):
        ui.label("NextAura v31 TriSplit GUI").classes("text-3xl font-black tracking-tight")
        ui.label("Stage 1 | Stage 2 | Stage 3").classes("text-sm text-slate-600")

        with ui.grid(columns=3).classes("w-full gap-4"):
            _pane("Stage 1", stage1_builder, "text-cyan-700")
            _pane("Stage 2", stage2_builder, "text-emerald-700")
            _pane("Stage 3", stage3_builder, "text-rose-700")
