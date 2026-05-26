from nicegui import app, ui
from gui.shared.layout import build_shell
from gui.stage1.view import stage1_view
from gui.stage2.view import stage2_view
from gui.stage3.view import stage3_view
from pathlib import Path

# Create static folder and recordings subfolder
static_dir = Path(__file__).resolve().parent / "static"
recordings_dir = static_dir / "recordings"
recordings_dir.mkdir(parents=True, exist_ok=True)

# Register static files directory
app.add_static_files('/static', str(static_dir))


@ui.page("/")
def home():
    build_shell(stage1_view, stage2_view, stage3_view)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="NextAura v31 GUI", reload=True)
