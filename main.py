from nicegui import ui
from gui.shared.layout import build_shell
from gui.stage1.view import stage1_view
from gui.stage2.view import stage2_view
from gui.stage3.view import stage3_view


@ui.page("/")
def home():
    build_shell(stage1_view, stage2_view, stage3_view)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="NextAura v31 GUI", reload=True)
