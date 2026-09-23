"""Compatibility launcher for ChatMol v2.

Recommended installation is the packaged ZIP from the repository's GitHub Actions
artifact or Release. This file remains for users who clone the repository and run:

    run /path/to/pymol_plugin/v2/chatmol.py
"""

import os
import sys

_PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from chatmol_ai.core import *  # noqa: F401,F403
from chatmol_ai.core import __init_plugin__  # noqa: F401

print("ChatMol compatibility launcher loaded. Package: chatmol_ai")
