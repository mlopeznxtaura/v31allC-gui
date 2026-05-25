from nicegui import ui


def stage2_view() -> None:
    ui.label("Stage 2 workspace").classes("text-sm text-slate-600")
    ui.input("Scenario").value = "stage2_placeholder"
    ui.slider(min=0, max=100, value=50).classes("w-full")
    ui.button("Run Stage 2")
    ui.label("Stage 2 status panel.").classes("text-xs text-slate-500")
