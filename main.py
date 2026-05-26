import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
gui_dir = Path(__file__).resolve().parent

# Remove script directory from sys.path to prevent shadowing of stage1/stage2/stage3
if str(gui_dir) in sys.path:
    sys.path.remove(str(gui_dir))

# Ensure project root is at sys.path[0]
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))

from nicegui import app, ui
from gui.shared.layout import build_shell
from gui.stage1.view import stage1_view
from gui.stage2.view import stage2_view
from gui.stage3.view import stage3_view



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
