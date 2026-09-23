#!/usr/bin/env python3
"""Build an installable PyMOL Plugin Manager ZIP for ChatMol AI."""

from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "pymol_plugin" / "chatmol_ai"
VERSION_FILE = PLUGIN_DIR / "version.py"
DIST = ROOT / "dist"

match = re.search(
    r'^__version__\s*=\s*["\']([^"\']+)["\']',
    VERSION_FILE.read_text(encoding="utf-8"),
    re.MULTILINE,
)
if not match:
    raise SystemExit("Could not read __version__ from version.py")

version = match.group(1)
DIST.mkdir(exist_ok=True)
out = DIST / f"chatmol-ai-v{version}.zip"

with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(PLUGIN_DIR.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rel = path.relative_to(PLUGIN_DIR)
        zf.write(path, Path("chatmol_ai") / rel)

print(out)
