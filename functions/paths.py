# functions/paths.py
from __future__ import annotations
from pathlib import Path
import sys

FROZEN = getattr(sys, "frozen", False)

def _candidate_roots():
    if FROZEN:
        # 1) EXE folder (onedir and onefile)
        yield Path(sys.executable).parent
        # 2) Onefile temp extraction dir
        mp = getattr(sys, "_MEIPASS", None)
        if mp:
            yield Path(mp)
    else:
        # 3) This file's folder (source)
        here = Path(__file__).resolve().parent
        yield here
        # 4) All ancestors (project roots)
        for p in here.parents:
            yield p

def _pick_app_dir() -> Path:
    for p in _candidate_roots():
        if (p / "pretrained").exists() or (p / "functions").exists():
            return p
    # last resort
    return Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent

# Public: base directory for assets/output (works in dev & frozen)
APP_DIR: Path = _pick_app_dir()

# If you need to walk upwards (replacement for Path(__file__).resolve().parents)
APP_PARENTS = list(APP_DIR.parents)

def resource_path(*parts: str) -> Path:
    """
    Prefer editable files next to the EXE (or project root),
    and fall back to the onefile extraction dir if bundled there.
    """
    # 1) External / editable path
    p = APP_DIR.joinpath(*parts)
    if p.exists():
        return p
    # 2) Onefile extraction fallback
    mp = getattr(sys, "_MEIPASS", None)
    if mp:
        q = Path(mp).joinpath(*parts)
        if q.exists():
            return q
    # 3) Return intended external path (caller may create it)
    return p

def find_project_root(markers=("pyproject.toml","requirements.txt",".git","functions","pretrained")) -> Path:
    """Walk up from APP_DIR looking for a root marker."""
    for base in [APP_DIR] + list(APP_PARENTS):
        for m in markers:
            if (base / m).exists():
                return base
    return APP_DIR
