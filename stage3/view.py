from nicegui import ui


def stage3_view() -> None:
    ui.label("Stage 3 workspace").classes("text-sm text-slate-600")
    ui.input("Deployment target").value = "stage3_placeholder"
    ui.switch("Dry run", value=True)
    ui.button("Run Stage 3")
    ui.label("Stage 3 status panel.").classes("text-xs text-slate-500")
