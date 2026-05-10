"""Session management: persists the active Colab endpoint URL across MCP server invocations.

Stored at `~/.openinterp/sessions.json`. NEVER contains secrets — only public ngrok URLs
and connection metadata (model_id, probes_loaded, last health-checked timestamp).

Multi-session is supported via named sessions. The "default" session is the one tools
operate on unless overridden.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional


SESSIONS_PATH = Path(os.environ.get("OPENINTERP_SESSIONS_PATH", "~/.openinterp/sessions.json")).expanduser()


@dataclass
class Session:
    name: str
    endpoint: str
    model_id: Optional[str] = None
    probes_loaded: int = 0
    captures_in_memory: int = 0
    attached_at: float = field(default_factory=time.time)
    last_health: float = 0.0


def _read_all() -> Dict[str, dict]:
    if not SESSIONS_PATH.exists():
        return {}
    try:
        return json.loads(SESSIONS_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _write_all(data: Dict[str, dict]) -> None:
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_PATH.write_text(json.dumps(data, indent=2))


def get_session(name: str = "default") -> Optional[Session]:
    data = _read_all()
    if name not in data:
        return None
    return Session(**data[name])


def put_session(session: Session) -> None:
    data = _read_all()
    data[session.name] = asdict(session)
    _write_all(data)


def remove_session(name: str = "default") -> bool:
    data = _read_all()
    if name not in data:
        return False
    del data[name]
    _write_all(data)
    return True


def list_sessions() -> Dict[str, Session]:
    return {name: Session(**body) for name, body in _read_all().items()}


def require_session(name: str = "default") -> Session:
    s = get_session(name)
    if not s:
        raise RuntimeError(
            f"No active session named `{name}`. Run `/colab-attach <ngrok-url>` first, "
            f"or check `~/.openinterp/sessions.json`."
        )
    return s
