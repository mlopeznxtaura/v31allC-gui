"""smart_recorder.py — Event-driven record builder
Only logs MEANINGFUL events, not padding/idle entries.
"""

import json
import time
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


class EventType(Enum):
    """Meaningful event types that should be recorded"""
    GUI_STARTUP = "gui_startup"
    USER_INPUT = "user_input"
    SCALAR_COMPUTE = "scalar_compute"
    TRIANGULATION_COMPUTE = "triangulation_compute"
    RECORD_EXPORT = "record_export"
    ERROR = "error"
    STATE_CHANGE = "state_change"
    GUI_BUTTON_CLICK = "gui_button_click"


@dataclass
class GUIEvent:
    """Single meaningful event"""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": round(self.timestamp, 3),
            "message": self.message,
            "data": self.data,
        }


@dataclass
class SmartRecorder:
    """
    Event-driven recorder. Only logs meaningful interactions.
    
    Usage:
        recorder = SmartRecorder(output_dir="/path/to/output")
        
        recorder.log_event(
            EventType.USER_INPUT,
            message="User entered stimulus value",
            data={"stimulus": 2.5, "M1": 1.234, "M2": 0.987}
        )
        
        recorder.save_session("my_run")
    """
    
    output_dir: Path = field(default_factory=lambda: Path("/mnt/d/NextAura/v31all_1/v31allC/output"))
    session_name: str = "default_session"
    events: List = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure output directory exists"""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event_type, message="", data=None):
        """Log a single meaningful event"""
        event = GUIEvent(
            event_type=event_type,
            message=message,
            data=data or {},
        )
        self.events.append(event)
        print(f"📌 [{event_type.value}] {message}")
    
    def log_scalar_tick(self, M1, M2, M3, V, stimulus):
        """Log M-scalar computation event"""
        self.log_event(
            EventType.SCALAR_COMPUTE,
            message=f"M-Scalars computed: M1={M1:.4f}, M2={M2:.4f}, M3={M3:.4f}",
            data={
                "M1": round(M1, 6),
                "M2": round(M2, 6),
                "M3": round(M3, 6),
                "V": round(V, 6),
                "stimulus": round(stimulus, 6),
            }
        )
    
    def log_record_built(self, entry_num, binary_text, geometry_text, language_text, output_tokens):
        """Log when a record is built"""
        self.log_event(
            EventType.RECORD_EXPORT,
            message=f"Record #{entry_num} built",
            data={
                "entry": entry_num,
                "binary_len": len(binary_text),
                "geometry_len": len(geometry_text),
                "language_len": len(language_text),
                "next_frame_prediction": output_tokens.get("next_frame_prediction", ""),
                "language_token_output": output_tokens.get("language_token_output", ""),
            }
        )
    
    def log_error(self, error_msg, exception=None):
        """Log an error event"""
        data = {"error": error_msg}
        if exception:
            data["exception_type"] = type(exception).__name__
            data["exception_str"] = str(exception)
        
        self.log_event(
            EventType.ERROR,
            message=error_msg,
            data=data,
        )
    
    def log_button_click(self, button_name, action):
        """Log GUI button click"""
        self.log_event(
            EventType.GUI_BUTTON_CLICK,
            message=f"Button clicked: {button_name}",
            data={"button": button_name, "action": action},
        )
    
    def save_session(self, filename=None):
        """Save all events to JSONL file"""
        if filename is None:
            filename = f"{self.session_name}_{int(time.time())}"
        
        output_path = self.output_dir / f"{filename}.jsonl"
        
        with open(output_path, "w") as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict()) + "\n")
        
        print(f"\n✅ Saved {len(self.events)} events to {output_path}")
        return output_path
    
    def summary(self):
        """Print event summary"""
        print(f"\n📊 Event Summary ({len(self.events)} total):")
        
        event_counts = {}
        for event in self.events:
            et = event.event_type.value
            event_counts[et] = event_counts.get(et, 0) + 1
        
        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type}: {count}")
