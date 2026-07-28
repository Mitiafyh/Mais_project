from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


HISTORY_FILE = Path(__file__).resolve().parent.parent / "diagnostic_history.json"


def load_history() -> list[dict[str, Any]]:
    """Load diagnostic history from JSON file."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_history(history: list[dict[str, Any]]) -> None:
    """Save diagnostic history to JSON file."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def add_diagnostic(
    image_name: str,
    prediction: str,
    confidence: float | None,
    model_name: str,
    image_path: str,
) -> None:
    """Add a new diagnostic entry to the history."""
    history = load_history()
    entry = {
        "id": len(history) + 1,
        "image_name": image_name,
        "prediction": prediction,
        "confidence": confidence,
        "model_name": model_name,
        "image_path": image_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    history.append(entry)
    save_history(history)


def get_history() -> list[dict[str, Any]]:
    """Get the full diagnostic history."""
    return load_history()