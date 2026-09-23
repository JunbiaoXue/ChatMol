"""Structured and raw PyMOL tools used by the ChatMol agent."""

import base64
import copy
import json
import os
import re
import tempfile
import time

from pymol import cmd

# 2. PyMOLTools — 4 tool definitions + execution
# ---------------------------------------------------------------------------


BLOCKED_COMMANDS = [
    ("blocked destructive command", re.compile(r"^\s*delete\s+all\s*$", re.I)),
    ("blocked destructive command", re.compile(r"^\s*remove\s+all\s*$", re.I)),
    ("blocked session reset command", re.compile(r"^\s*reinitialize(?:\s|$)", re.I)),
    ("blocked quit command", re.compile(r"^\s*quit(?:\s|$)", re.I)),
    ("blocked script execution command", re.compile(r"^\s*run(?:\s|$)", re.I)),
    ("blocked python command", re.compile(r"^\s*python(?:\s|$)", re.I)),
    (
        "blocked shell/system command",
        re.compile(
            r"^\s*(?:!|shell(?:\s|$)|system(?:\s|$)|os\.system|subprocess\.)", re.I
        ),
    ),
]

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_session",
            "description": (
                "Inspect the current PyMOL session. Returns structured JSON "
                "with objects, chain summaries, atom counts, and selections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_view": {
                        "type": "boolean",
                        "description": "Include camera view matrix.",
                        "default": False,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pymol_commands",
            "description": (
                "Execute newline-separated PyMOL commands. Use this for all "
                "PyMOL operations: selections, styling, coloring, distances, "
                "presets, etc. Commands are run sequentially with safety "
                "guardrails (destructive/shell commands are blocked)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "string",
                        "description": "Newline-separated PyMOL commands.",
                    }
                },
                "required": ["commands"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render",
            "description": (
                "Render and export an image from the current scene. "
                "Use purpose=preview for quick iterative checks, and "
                "purpose=final for final publication export."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output image path.",
                    },
                    "purpose": {
                        "type": "string",
                        "enum": ["preview", "final"],
                        "default": "preview",
                        "description": "preview is lightweight; final is high quality.",
                    },
                    "width": {"type": "integer", "default": 1400},
                    "height": {"type": "integer", "default": 1050},
                    "dpi": {"type": "integer", "default": 150},
                    "ray": {
                        "type": "boolean",
                        "description": "If omitted: preview=False, final=True.",
                    },
                    "transparent_bg": {"type": "boolean", "default": True},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_viewport",
            "description": (
                "Capture current viewport and run vision model analysis. "
                "Use this to visually verify your work (e.g., check that "
                "styling looks correct before finalizing)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_prompt": {
                        "type": "string",
                        "default": "Describe what you see in this molecular visualization.",
                    }
                },
            },
        },
    },
]



# Prefer these structured tools over raw command strings whenever possible.
TOOL_DEFINITIONS += [
    {
        "type": "function",
        "function": {
            "name": "select_residues",
            "description": "Create or replace a named PyMOL selection from a selection expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection": {"type": "string"},
                    "name": {"type": "string", "default": "chatmol_selection"},
                },
                "required": ["selection"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "color_selection",
            "description": "Color a PyMOL selection with a named PyMOL color.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection": {"type": "string"},
                    "color": {"type": "string"},
                },
                "required": ["selection", "color"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_representation",
            "description": "Show a representation such as cartoon, sticks, surface, spheres, or lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection": {"type": "string"},
                    "representation": {"type": "string"},
                },
                "required": ["selection", "representation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_distance",
            "description": "Create a PyMOL distance object between two selections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection1": {"type": "string"},
                    "selection2": {"type": "string"},
                    "name": {"type": "string", "default": "chatmol_distance"},
                },
                "required": ["selection1", "selection2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "align_structures",
            "description": "Align a mobile selection/object onto a target selection/object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mobile": {"type": "string"},
                    "target": {"type": "string"},
                    "cycles": {"type": "integer", "default": 5},
                },
                "required": ["mobile", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sequence",
            "description": "Return FASTA sequence for a PyMOL selection without changing the scene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection": {"type": "string", "default": "polymer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contacts",
            "description": "Return atom index pairs within a cutoff between two selections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selection1": {"type": "string"},
                    "selection2": {"type": "string"},
                    "cutoff": {"type": "number", "default": 4.0},
                },
                "required": ["selection1", "selection2"],
            },
        },
    },
]

MUTATING_TOOLS = {
    "run_pymol_commands",
    "select_residues",
    "color_selection",
    "show_representation",
    "measure_distance",
    "align_structures",
}

_SESSION_HISTORY = []
_SESSION_HISTORY_LIMIT = 10


def is_mutating_tool(tool_name):
    return tool_name in MUTATING_TOOLS


def preview_tool_call(tool_name, arguments):
    return json.dumps(
        {
            "ok": True,
            "tool": tool_name,
            "executed": False,
            "dry_run": True,
            "arguments": arguments or {},
            "message": "Dry Run: action was planned but not executed. Do not assume the PyMOL state changed.",
        },
        ensure_ascii=False,
    )


def snapshot_session(label="AI action"):
    """Keep an in-memory PyMOL session snapshot for one-step/multi-step undo."""
    try:
        session = copy.deepcopy(cmd.get_session())
        _SESSION_HISTORY.append((label, session))
        if len(_SESSION_HISTORY) > _SESSION_HISTORY_LIMIT:
            del _SESSION_HISTORY[0]
        return {"ok": True, "label": label, "depth": len(_SESSION_HISTORY)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "depth": len(_SESSION_HISTORY)}


def undo_last_action():
    """Restore the most recent snapshot created before an AI mutation."""
    if not _SESSION_HISTORY:
        return {"ok": False, "error": "No ChatMol undo snapshot is available.", "depth": 0}
    label, session = _SESSION_HISTORY.pop()
    try:
        cmd.set_session(session)
        return {"ok": True, "restored": label, "depth": len(_SESSION_HISTORY)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "depth": len(_SESSION_HISTORY)}


def _tool_select_residues(selection, name="chatmol_selection"):
    count = cmd.select(name, selection)
    return {"ok": True, "tool": "select_residues", "name": name, "selection": selection, "atoms": count}


def _tool_color_selection(selection, color):
    cmd.color(color, selection)
    return {"ok": True, "tool": "color_selection", "selection": selection, "color": color}


def _tool_show_representation(selection, representation):
    cmd.show(representation, selection)
    return {"ok": True, "tool": "show_representation", "selection": selection, "representation": representation}


def _tool_measure_distance(selection1, selection2, name="chatmol_distance"):
    value = cmd.distance(name, selection1, selection2)
    return {"ok": True, "tool": "measure_distance", "name": name, "distance": value}


def _tool_align_structures(mobile, target, cycles=5):
    result = cmd.align(mobile, target, cycles=int(cycles))
    return {"ok": True, "tool": "align_structures", "mobile": mobile, "target": target, "result": list(result) if isinstance(result, tuple) else result}


def _tool_get_sequence(selection="polymer"):
    return {"ok": True, "tool": "get_sequence", "selection": selection, "fasta": cmd.get_fastastr(selection)}


def _tool_get_contacts(selection1, selection2, cutoff=4.0):
    pairs = cmd.find_pairs(selection1, selection2, cutoff=float(cutoff), mode=0)
    return {"ok": True, "tool": "get_contacts", "selection1": selection1, "selection2": selection2, "cutoff": float(cutoff), "count": len(pairs), "pairs": pairs[:500]}


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    if value is None:
        return default
    return bool(value)


def _blocked_reason(line):
    for reason, pattern in BLOCKED_COMMANDS:
        if pattern.search(line):
            return reason
    return ""


def execute_tool(tool_name, arguments, client=None, vision_model=None):
    """Dispatch to a tool implementation and always return JSON text."""
    dispatch = {
        "inspect_session": _tool_inspect_session,
        "run_pymol_commands": _tool_run_pymol_commands,
        "render": _tool_render,
        "capture_viewport": _tool_capture_viewport,
        "select_residues": _tool_select_residues,
        "color_selection": _tool_color_selection,
        "show_representation": _tool_show_representation,
        "measure_distance": _tool_measure_distance,
        "align_structures": _tool_align_structures,
        "get_sequence": _tool_get_sequence,
        "get_contacts": _tool_get_contacts,
    }
    fn = dispatch.get(tool_name)
    if fn is None:
        return json.dumps(
            {"ok": False, "tool": tool_name, "error": f"Unknown tool: {tool_name}"},
            ensure_ascii=False,
        )
    snapshot = None
    if tool_name in MUTATING_TOOLS:
        snapshot = snapshot_session(tool_name)
    try:
        if tool_name == "capture_viewport":
            result = fn(client=client, vision_model=vision_model, **(arguments or {}))
        else:
            result = fn(**(arguments or {}))
    except TypeError as exc:
        result = {"ok": False, "tool": tool_name, "error": f"Invalid arguments: {exc}"}
    except Exception as exc:
        result = {"ok": False, "tool": tool_name, "error": str(exc)}
    if isinstance(result, dict) and snapshot is not None:
        result["undo_snapshot"] = bool(snapshot.get("ok"))
        result["undo_depth"] = snapshot.get("depth", 0)
    return json.dumps(result, ensure_ascii=False)


def _tool_inspect_session(include_view=False):
    objects = cmd.get_names("objects")
    selections = cmd.get_names("selections")
    details = []
    for obj in objects:
        details.append(
            {
                "name": obj,
                "atoms": cmd.count_atoms(obj),
                "polymer_atoms": cmd.count_atoms(f"({obj}) and polymer"),
                "hetatm_atoms": cmd.count_atoms(f"({obj}) and hetatm"),
                "chains": cmd.get_chains(obj),
                "states": cmd.count_states(obj),
            }
        )
    out = {
        "ok": True,
        "tool": "inspect_session",
        "object_count": len(objects),
        "selection_count": len(selections),
        "objects": details,
        "selections": selections,
    }
    if _to_bool(include_view):
        out["view"] = list(cmd.get_view())
    return out


def _tool_run_pymol_commands(commands):
    lines = [
        ln.strip()
        for ln in commands.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    executed, blocked, errors = [], [], []
    known_names = set(cmd.get_names("all"))
    for line in lines:
        reason = _blocked_reason(line)
        if reason:
            blocked.append({"command": line, "reason": reason})
            continue
        m_del = re.match(r"^\s*delete\s+([A-Za-z0-9_]+)\s*$", line, re.I)
        if m_del:
            target = m_del.group(1)
            if target not in known_names:
                continue  # skip silently
        try:
            cmd.do(line)
            executed.append(line)
            if m_del:
                known_names.discard(m_del.group(1))
        except Exception as exc:
            errors.append({"command": line, "error": str(exc)})
    return {
        "ok": not blocked and not errors,
        "tool": "run_pymol_commands",
        "executed_count": len(executed),
        "blocked_count": len(blocked),
        "error_count": len(errors),
        "executed": executed,
        "blocked": blocked,
        "errors": errors,
    }


def _tool_render(
    path,
    width=None,
    height=None,
    dpi=None,
    ray=None,
    transparent_bg=True,
    purpose="preview",
):
    out_path = os.path.abspath(os.path.expanduser(path))
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    purpose = (purpose or "preview").strip().lower()
    if purpose not in ("preview", "final"):
        purpose = "preview"

    if purpose == "final":
        default_width, default_height, default_dpi = 2400, 1800, 300
        default_ray = True
    else:
        default_width, default_height, default_dpi = 1400, 1050, 150
        default_ray = False

    try:
        width = default_width if width is None else int(width)
        height = default_height if height is None else int(height)
        dpi = default_dpi if dpi is None else int(dpi)
        width = max(256, width)
        height = max(256, height)
        dpi = max(72, dpi)
    except (TypeError, ValueError):
        width, height, dpi = default_width, default_height, default_dpi

    if purpose == "preview":
        width = min(width, 2200)
        height = min(height, 1650)
        dpi = min(dpi, 220)

    ray_flag = 1 if _to_bool(ray, default=default_ray) else 0
    transparent = _to_bool(transparent_bg, default=True)
    cmd.set("ray_opaque_background", 0 if transparent else 1)
    cmd.png(out_path, width=width, height=height, dpi=dpi, ray=ray_flag, quiet=1)

    exists = os.path.exists(out_path)
    size = os.path.getsize(out_path) if exists else 0
    return {
        "ok": exists,
        "tool": "render",
        "path": out_path,
        "purpose": purpose,
        "width": width,
        "height": height,
        "dpi": dpi,
        "ray": bool(ray_flag),
        "transparent_bg": transparent,
        "bytes": size,
    }


def _tool_capture_viewport(analysis_prompt=None, client=None, vision_model=None):
    if analysis_prompt is None:
        analysis_prompt = "Describe what you see in this molecular visualization."

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        cmd.png(tmp_path, width=800, height=600, ray=0, quiet=1)
        time.sleep(0.5)
        with open(tmp_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("ascii")
    except Exception as exc:
        return {
            "ok": False,
            "tool": "capture_viewport",
            "error": f"Failed to capture viewport: {exc}",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not vision_model or not client:
        return {
            "ok": False,
            "tool": "capture_viewport",
            "error": "No vision model configured. Use set_vision_model <model>.",
        }

    try:
        description = client.vision_completion(vision_model, analysis_prompt, img_data)
        return {
            "ok": True,
            "tool": "capture_viewport",
            "analysis_prompt": analysis_prompt,
            "description": description,
        }
    except Exception as exc:
        return {
            "ok": False,
            "tool": "capture_viewport",
            "error": f"Vision analysis failed: {exc}",
        }


# ---------------------------------------------------------------------------
