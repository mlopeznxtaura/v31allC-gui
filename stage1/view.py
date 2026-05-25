from nicegui import ui


def stage1_view() -> None:
    ui.label("Layer-1 + Layer-2 + Triangulation").classes("text-sm text-slate-600")
    ui.input("Mission").value = "stage1_gui_boot"
    ui.number("Stimulus", value=1.0, format="%.2f")
    ui.button("Run Stage 1 Tick")
    ui.label("Output preview will render here.").classes("text-xs text-slate-500")
