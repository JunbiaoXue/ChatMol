"""ChatMol agent, Qt UI, and PyMOL plugin entry points."""

import json
import os
import random
import threading
import time
from datetime import datetime

import requests
from pymol import cmd

from .providers import PROVIDERS, LLMClient, LLMTransientError
from .tools import (
    TOOL_DEFINITIONS,
    execute_tool,
    is_mutating_tool,
    preview_tool_call,
    undo_last_action,
)
from .version import __version__

# 3. System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are ChatMol, an expert AI assistant for PyMOL molecular visualization.

You have structured PyMOL tools plus a raw-command fallback:
- `inspect_session`, `get_sequence`, `get_contacts`: inspect without changing the scene.
- `select_residues`, `color_selection`, `show_representation`, `measure_distance`, `align_structures`: preferred structured editing tools.
- `run_pymol_commands`: fallback for PyMOL operations not covered by structured tools.
- `render`: export an image.
- `capture_viewport`: screenshot + vision analysis for visual QA.

Prefer structured tools whenever they can express the requested operation. Use raw commands only when needed.
If a tool result says dry_run, confirmation_required, or denied, do not assume the PyMOL state changed; stop mutating and summarize the planned action.

Workflow:
1. For non-trivial requests, start with `inspect_session` to understand the current state.
2. Use `run_pymol_commands` for all PyMOL operations. You know PyMOL well — use \
cmd.select, cmd.show, cmd.hide, cmd.color, cmd.set, cmd.distance, cmd.zoom, \
cmd.orient, util.color_chains, util.cnc, preset.ligand_sites_hq, etc.
3. Use `capture_viewport` to verify your work visually when doing complex styling.
4. Use `render` when the user requests an image export.

Style guidelines (publication quality):
- White background, clean composition.
- Cartoon as baseline representation for protein.
- Sticks only for key residues, ligands, and interaction sites.
- Context in gray; 1-2 accent colors (often cyan/marine + orange).
- Surfaces used purposefully for targets/interfaces/pores, not everywhere.
- Hydrogen bonds / polar contacts shown with dashed lines when relevant.
- Keep decoration sparse — prioritize clarity over ornamentation.

Rules:
- Never issue destructive commands (reinitialize, quit, delete all, shell commands).
- If user input is ambiguous, ask for clarification rather than guessing.
- Be concise in your final response.
"""


# ---------------------------------------------------------------------------
# 4. ChatMolAgent — config persistence + simple agent loop
# ---------------------------------------------------------------------------


class ChatMolAgent:
    """Simple agentic loop: call LLM, execute tools, repeat."""

    CONFIG_PATH = os.path.expanduser("~/.PyMOL/chatmol_config.json")

    DEFAULT_CONFIG = {
        "provider": "openrouter",
        "api_keys": {},
        "base_urls": {},
        "text_model": "google/gemini-3-flash-preview",
        "vision_model": "google/gemini-3-flash-preview",
        "execution_mode": "auto",
        "temperature": 0.01,
        "max_tokens": 4096,
        "max_iterations": 50,
        "max_tool_calls": 30,
    }

    def __init__(self):
        os.makedirs(os.path.dirname(self.CONFIG_PATH), exist_ok=True)
        self.config = self._load_config()
        self._reinit_client()
        self.conversation_history = []

    # -- config persistence -------------------------------------------------

    def _load_config(self):
        try:
            with open(self.CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            merged = dict(self.DEFAULT_CONFIG)
            merged["api_keys"] = dict(self.DEFAULT_CONFIG["api_keys"])
            merged["base_urls"] = dict(self.DEFAULT_CONFIG["base_urls"])
            merged.update(cfg)
            if not isinstance(merged.get("api_keys"), dict):
                merged["api_keys"] = {}
            if not isinstance(merged.get("base_urls"), dict):
                merged["base_urls"] = {}
            # Migrate old single "api_key" field
            old_key = merged.pop("api_key", "")
            if old_key and isinstance(old_key, str):
                prov = merged.get("provider", "openrouter")
                merged["api_keys"].setdefault(prov, old_key)
                self._save_config(merged)
            # Remove legacy keys
            for legacy in ("human_in_the_loop", "multi_agent_enabled"):
                merged.pop(legacy, None)
            return merged
        except (FileNotFoundError, json.JSONDecodeError):
            self._save_config(self.DEFAULT_CONFIG)
            return dict(self.DEFAULT_CONFIG)

    def _save_config(self, cfg=None):
        if cfg is None:
            cfg = self.config
        try:
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as exc:
            print(f"Warning: could not save config: {exc}")

    def _resolve_api_key(self):
        prov_name = self.config.get("provider", "openrouter")
        prov = PROVIDERS.get(prov_name, PROVIDERS["openrouter"])
        return os.getenv(prov["env_var"], "") or self.config.get("api_keys", {}).get(
            prov_name, ""
        )

    def _resolve_base_url(self):
        prov_name = self.config.get("provider", "openrouter")
        prov = PROVIDERS.get(prov_name, PROVIDERS["openrouter"])
        stored = self.config.get("base_urls", {}).get(prov_name, "")
        return stored.strip() or prov.get("base_url", "")

    def _reinit_client(self):
        prov_name = self.config.get("provider", "openrouter")
        api_key = self._resolve_api_key()
        self.client = LLMClient(prov_name, api_key, self._resolve_base_url())

    def fetch_available_models(self):
        """Fetch model IDs from the current provider's OpenAI-compatible /models endpoint."""
        self._reinit_client()
        return self.client.list_models()

    # -- PyMOL-registered commands ------------------------------------------

    def chat(self, *args):
        """PyMOL command: chat <message>"""
        message = " ".join(str(a) for a in args).strip()
        if not message:
            print("Usage: chat <message>")
            return
        if not self._resolve_api_key():
            prov = self.config.get("provider", "openrouter")
            env = PROVIDERS.get(prov, {}).get("env_var", "")
            print(f"No API key set. Use: set_api_key <key>  (or set env var {env})")
            return
        if not self._resolve_base_url():
            print("No Base URL set. Use: set_base_url <url>  (for example https://host/v1)")
            return
        if not self.config.get("text_model", "").strip():
            print("No text model set. Use: set_model <model-id>")
            return
        try:
            response = self._run_agent_loop(message)
            print("=" * 50)
            print("ChatMol:", response)
            print("=" * 50)
        except RuntimeError as exc:
            print(f"ChatMol error: {exc}")
        except requests.exceptions.RequestException as exc:
            print(f"ChatMol network error: {exc}")
        except (KeyError, IndexError, TypeError) as exc:
            prov = self.config.get("provider", "openrouter")
            model = self.config.get("text_model", "")
            print(f"ChatMol error: unexpected response format — {exc}")
            print(f"  Provider: {prov}, Model: {model}")
            print(
                "  The model may not support tool calling, "
                "or the API returned an unexpected structure."
            )
        except Exception as exc:
            print(f"ChatMol error ({type(exc).__name__}): {exc}")

    def set_api_key(self, api_key=""):
        """PyMOL command: set_api_key <key>"""
        api_key = api_key.strip()
        if not api_key:
            prov_name = self.config.get("provider", "openrouter")
            label = PROVIDERS.get(prov_name, {}).get("label", prov_name)
            print(f"Usage: set_api_key <your_{label}_api_key>")
            return
        prov_name = self.config.get("provider", "openrouter")
        if not isinstance(self.config.get("api_keys"), dict):
            self.config["api_keys"] = {}
        self.config["api_keys"][prov_name] = api_key
        self._save_config()
        self._reinit_client()
        label = PROVIDERS.get(prov_name, {}).get("label", prov_name)
        print(f"{label} API key saved.")

    def set_base_url(self, base_url=""):
        """PyMOL command: set_base_url <url>"""
        prov_name = self.config.get("provider", "openrouter")
        base_url = base_url.strip()
        if not base_url:
            current = self._resolve_base_url() or "(not set)"
            print(f"Current Base URL: {current}")
            print("Usage: set_base_url https://your-newapi.example/v1")
            return
        if not isinstance(self.config.get("base_urls"), dict):
            self.config["base_urls"] = {}
        self.config["base_urls"][prov_name] = base_url.rstrip("/")
        self._save_config()
        self._reinit_client()
        print(f"Base URL for {prov_name} set to: {self.config['base_urls'][prov_name]}")
        print(f"Chat completions endpoint: {self.client.base_url}")

    def set_provider(self, provider_name=""):
        """PyMOL command: set_provider <name>"""
        provider_name = provider_name.strip().lower()
        if not provider_name or provider_name not in PROVIDERS:
            print(f"Available providers: {', '.join(PROVIDERS.keys())}")
            if provider_name:
                print(f"Unknown provider: {provider_name}")
            return
        self.config["provider"] = provider_name
        prov = PROVIDERS[provider_name]
        if prov["models"]:
            self.config["text_model"] = prov["models"][0][0]
        if prov["vision_models"]:
            self.config["vision_model"] = prov["vision_models"][0][0]
        elif provider_name != "newapi":
            self.config["vision_model"] = ""
        self._save_config()
        self._reinit_client()
        print(f"Provider set to: {prov['label']}")
        print(f"  Text model: {self.config['text_model']}")
        if self.config["vision_model"]:
            print(f"  Vision model: {self.config['vision_model']}")
        print(f"  Set API key via: set_api_key <key>  (or env {prov['env_var']})")

    def set_model(self, model_name=""):
        """PyMOL command: set_model <model>"""
        model_name = model_name.strip()
        if not model_name:
            print(f"Current model: {self.config['text_model']}")
            print("Usage: set_model <provider/model>")
            return
        self.config["text_model"] = model_name
        self._save_config()
        print(f"Text model set to: {model_name}")

    def set_vision_model(self, model_name=""):
        """PyMOL command: set_vision_model <model>"""
        model_name = model_name.strip()
        if not model_name:
            print(f"Current vision model: {self.config['vision_model']}")
            print("Usage: set_vision_model <provider/model>")
            return
        self.config["vision_model"] = model_name
        self._save_config()
        print(f"Vision model set to: {model_name}")

    def reset_conversation(self):
        """PyMOL command: reset_conversation"""
        self.conversation_history = []
        print("Conversation history cleared.")

    def save_conversation(self, filename=""):
        """PyMOL command: save_conversation [filename]"""
        if not filename:
            filename = (
                f"chatmol_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=2)
        print(f"Conversation saved to {filename}")

    def load_conversation(self, filename=""):
        """PyMOL command: load_conversation <filename>"""
        if not filename:
            print("Usage: load_conversation <filename.json>")
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.conversation_history = json.load(f)
            print(
                f"Loaded conversation from {filename} "
                f"({len(self.conversation_history)} messages)"
            )
        except Exception as exc:
            print(f"Failed to load conversation: {exc}")

    def show_config(self):
        """PyMOL command: chatmol_config"""
        prov_name = self.config.get("provider", "openrouter")
        prov = PROVIDERS.get(prov_name, {})
        key = self._resolve_api_key()
        masked = (key[:8] + "..." + key[-4:]) if key and len(key) > 12 else "(not set)"
        has_env = bool(os.getenv(prov.get("env_var", ""), ""))
        print("ChatMol configuration:")
        print(f"  provider: {prov.get('label', prov_name)} ({prov_name})")
        print(f"  base_url: {self._resolve_base_url() or '(not set)'}")
        print(f"  endpoint: {self.client.base_url or '(not set)'}")
        print(f"  api_key: {masked} ({'env var' if has_env else 'config file'})")
        print(f"  text_model: {self.config.get('text_model', '')}")
        print(f"  vision_model: {self.config.get('vision_model', '') or '(none)'}")
        print(f"  execution_mode: {self.config.get('execution_mode', 'auto')}")
        print(f"  version: {__version__}")
        print(f"  temperature: {self.config.get('temperature', 0.01)}")
        print(f"  max_tokens: {self.config.get('max_tokens', 4096)}")
        print(f"  max_iterations: {self.config.get('max_iterations', 50)}")
        print(f"  max_tool_calls: {self.config.get('max_tool_calls', 30)}")
        stored = [k for k, v in self.config.get("api_keys", {}).items() if v]
        if stored:
            print(f"  keys stored for: {', '.join(stored)}")

    # -- agentic loop -------------------------------------------------------

    def _chat_completion_with_retry(
        self, model, messages, tool_defs, temperature, max_tokens, phase_callback=None
    ):
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                return self.client.chat_completion(
                    model,
                    messages,
                    tools=tool_defs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except LLMTransientError as exc:
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Transient API errors persisted after {max_attempts} attempts: {exc}"
                    )
                delay = min(8.0, float(2 ** (attempt - 1))) + random.uniform(0.0, 0.4)
                print(
                    f"  [Retry {attempt}/{max_attempts}] transient error: "
                    f"{exc}. Sleeping {delay:.1f}s."
                )
                if phase_callback:
                    phase_callback("Retrying API request")
                time.sleep(delay)

    @staticmethod
    def _parse_assistant_message(data):
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Invalid API response: missing choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Invalid API response: missing assistant message.")
        return message

    @staticmethod
    def _normalize_content(content):
        if isinstance(content, (list, dict)):
            return json.dumps(content, ensure_ascii=False)
        if content is None:
            return ""
        if not isinstance(content, str):
            return str(content)
        return content

    @staticmethod
    def _parse_tool_calls(assistant_msg):
        raw_calls = assistant_msg.get("tool_calls") or []
        parsed = []
        for tc in raw_calls:
            if not isinstance(tc, dict):
                raise RuntimeError("Invalid tool call entry from model.")
            tc_id = tc.get("id")
            fn = tc.get("function") or {}
            fn_name = fn.get("name")
            raw_args = fn.get("arguments", "{}")
            if not tc_id or not fn_name:
                raise RuntimeError(
                    "Invalid tool call payload (missing id or function name)."
                )
            if isinstance(raw_args, dict):
                fn_args = raw_args
            else:
                raw_args = "{}" if raw_args is None else str(raw_args).strip()
                if not raw_args:
                    raw_args = "{}"
                try:
                    fn_args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Tool arguments not valid JSON for {fn_name}: {exc}"
                    )
            if not isinstance(fn_args, dict):
                raise RuntimeError(f"Tool arguments for {fn_name} must be an object.")
            parsed.append(
                {"id": tc_id, "name": fn_name, "arguments": fn_args, "raw": tc}
            )
        return parsed

    def _make_tool_executor(self, tool_executor_override=None):
        """Return a tool executor function that passes client/vision_model."""
        if tool_executor_override:
            return tool_executor_override
        client = self.client
        vision_model = self.config.get("vision_model", "")

        execution_mode = self.config.get("execution_mode", "auto")

        def _executor(tool_name, arguments):
            if is_mutating_tool(tool_name):
                if execution_mode == "dry_run":
                    return preview_tool_call(tool_name, arguments)
                if execution_mode == "confirm":
                    return json.dumps(
                        {
                            "ok": False,
                            "tool": tool_name,
                            "confirmation_required": True,
                            "executed": False,
                            "message": (
                                "Confirm mode requires the ChatMol GUI. "
                                "Use the GUI to approve this action or switch execution mode to Auto."
                            ),
                            "arguments": arguments or {},
                        },
                        ensure_ascii=False,
                    )
            return execute_tool(
                tool_name, arguments, client=client, vision_model=vision_model
            )

        return _executor

    def _run_agent_loop_internal(
        self, message, tool_executor=None, phase_callback=None
    ):
        self.conversation_history.append({"role": "user", "content": message})

        model = self.config["text_model"]
        temperature = self.config.get("temperature", 0.01)
        max_tokens = self.config.get("max_tokens", 4096)
        max_iterations = max(1, int(self.config.get("max_iterations", 50) or 50))
        max_tool_calls = max(8, int(self.config.get("max_tool_calls", 30) or 30))
        executor = self._make_tool_executor(tool_executor)
        tool_calls_used = 0

        for iteration in range(max_iterations):
            if phase_callback:
                phase_callback("Thinking" if iteration == 0 else "Thinking more")

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages += self.conversation_history

            data = self._chat_completion_with_retry(
                model,
                messages,
                TOOL_DEFINITIONS,
                temperature,
                max_tokens,
                phase_callback,
            )
            assistant_msg = self._parse_assistant_message(data)
            parsed_calls = self._parse_tool_calls(assistant_msg)
            content = self._normalize_content(assistant_msg.get("content", None))

            history_entry = {"role": "assistant", "content": content}
            if parsed_calls:
                history_entry["tool_calls"] = [tc["raw"] for tc in parsed_calls]

            if parsed_calls:
                tool_entries = []
                for tc in parsed_calls:
                    fn_name = tc["name"]
                    fn_args = tc["arguments"]

                    if phase_callback:
                        phase_callback(f"Running {fn_name}")
                    print(
                        f"  [Tool: {fn_name}] "
                        f"{json.dumps(fn_args, ensure_ascii=False)[:120]}"
                    )

                    if tool_calls_used >= max_tool_calls:
                        result = json.dumps(
                            {
                                "ok": False,
                                "tool": fn_name,
                                "error": (
                                    "Tool-call budget reached. Stop calling tools "
                                    "and provide your final response."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    else:
                        try:
                            result = executor(fn_name, fn_args)
                        except Exception as exc:
                            result = json.dumps(
                                {"ok": False, "tool": fn_name, "error": str(exc)},
                                ensure_ascii=False,
                            )

                    tool_entries.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )
                    tool_calls_used += 1

                self.conversation_history.append(history_entry)
                self.conversation_history.extend(tool_entries)
                continue

            # No tool calls — this is the final response
            self.conversation_history.append(history_entry)
            return (content or "").strip()

        return (
            "Agent loop reached maximum iterations "
            f"({max_iterations}) without a final response."
        )

    def _run_agent_loop(self, message):
        return self._run_agent_loop_internal(message, tool_executor=None)

    def run_agent_loop_with_callback(self, message, tool_executor, phase_callback=None):
        return self._run_agent_loop_internal(
            message, tool_executor=tool_executor, phase_callback=phase_callback
        )


# ---------------------------------------------------------------------------
# 5. Qt5 GUI (loaded conditionally)
# ---------------------------------------------------------------------------

_HAS_QT = False
try:
    from pymol.Qt import QtWidgets, QtCore, QtGui  # noqa: E402

    QDockWidget = QtWidgets.QDockWidget
    QWidget = QtWidgets.QWidget
    QVBoxLayout = QtWidgets.QVBoxLayout
    QHBoxLayout = QtWidgets.QHBoxLayout
    QTextEdit = QtWidgets.QTextEdit
    QLineEdit = QtWidgets.QLineEdit
    QPushButton = QtWidgets.QPushButton
    QLabel = QtWidgets.QLabel
    QDialog = QtWidgets.QDialog
    QFormLayout = QtWidgets.QFormLayout
    QDoubleSpinBox = QtWidgets.QDoubleSpinBox
    QSpinBox = QtWidgets.QSpinBox
    QComboBox = QtWidgets.QComboBox
    QApplication = QtWidgets.QApplication

    Qt = QtCore.Qt
    QThread = QtCore.QThread
    QTimer = QtCore.QTimer
    QPropertyAnimation = QtCore.QPropertyAnimation
    QEasingCurve = QtCore.QEasingCurve
    Signal = getattr(QtCore, "Signal", None) or getattr(QtCore, "pyqtSignal")
    Property = getattr(QtCore, "Property", None) or getattr(QtCore, "pyqtProperty")

    QColor = QtGui.QColor
    QPainter = QtGui.QPainter

    _HAS_QT = True
except ImportError:
    pass


if _HAS_QT:

    # -- ThinkingIndicator --------------------------------------------------

    class ThinkingIndicator(QWidget):
        """Three dots that pulse with a breathing animation."""

        _DOT_COUNT = 3
        _DOT_RADIUS = 4
        _DOT_SPACING = 14

        def __init__(self, parent=None):
            super().__init__(parent)
            self._opacity = 0.3
            self._phase_text = "Thinking"
            self.setFixedHeight(28)
            self._anim = QPropertyAnimation(self, b"dot_opacity")
            self._anim.setDuration(900)
            self._anim.setStartValue(0.3)
            self._anim.setEndValue(1.0)
            self._anim.setEasingCurve(QEasingCurve.InOutSine)
            self._anim.setLoopCount(-1)
            self._anim.finished.connect(lambda: None)
            self._forward = True
            self._cycle_timer = QTimer(self)
            self._cycle_timer.setInterval(900)
            self._cycle_timer.timeout.connect(self._toggle_direction)
            self.hide()

        def _toggle_direction(self):
            self._forward = not self._forward
            if self._forward:
                self._anim.setStartValue(0.3)
                self._anim.setEndValue(1.0)
            else:
                self._anim.setStartValue(1.0)
                self._anim.setEndValue(0.3)
            self._anim.start()

        @Property(float)
        def dot_opacity(self):
            return self._opacity

        @dot_opacity.setter
        def dot_opacity(self, value):
            self._opacity = value
            self.update()

        def start(self, phase_text="Thinking"):
            self._phase_text = phase_text
            self._anim.setStartValue(0.3)
            self._anim.setEndValue(1.0)
            self._forward = True
            self._anim.start()
            self._cycle_timer.start()
            self.show()

        def stop(self):
            self._anim.stop()
            self._cycle_timer.stop()
            self.hide()

        def set_phase(self, text):
            self._phase_text = text
            self.update()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QColor(120, 120, 120))
            p.drawText(8, 18, self._phase_text)
            label_width = p.fontMetrics().horizontalAdvance(self._phase_text) + 16
            color = QColor(70, 130, 230)
            for i in range(self._DOT_COUNT):
                offset = i * 0.15
                alpha = max(0.0, min(1.0, self._opacity - offset))
                color.setAlphaF(alpha)
                p.setBrush(color)
                p.setPen(Qt.NoPen)
                x = label_width + i * self._DOT_SPACING
                p.drawEllipse(x, 10, self._DOT_RADIUS * 2, self._DOT_RADIUS * 2)
            p.end()

    # -- AgentWorker --------------------------------------------------------

    class AgentWorker(QThread):
        """Runs the agentic loop off the main thread."""

        finished = Signal(str)
        error = Signal(str)
        tool_executed = Signal(str, str)
        request_tool_execution = Signal(str, str, str)
        phase_changed = Signal(str)

        def __init__(self, agent, message, parent=None):
            super().__init__(parent)
            self.agent = agent
            self.message = message
            self._tool_result = None
            self._tool_event = threading.Event()

        def run(self):
            try:
                response = self.agent.run_agent_loop_with_callback(
                    self.message, self._tool_executor, phase_callback=self._on_phase
                )
                self.finished.emit(response)
            except Exception as exc:
                self.error.emit(str(exc))

        def _on_phase(self, phase_text):
            self.phase_changed.emit(phase_text)

        def _tool_executor(self, tool_name, arguments):
            self._tool_event.clear()
            self._tool_result = None
            args_json = json.dumps(arguments, ensure_ascii=False)
            self.request_tool_execution.emit("", tool_name, args_json)
            self._tool_event.wait(timeout=60)
            result = self._tool_result or "Tool execution timed out."
            self.tool_executed.emit(tool_name, result[:200])
            return result

        def deliver_tool_result(self, result):
            self._tool_result = result
            self._tool_event.set()

    # -- ChatMolSettingsDialog ----------------------------------------------

    class ChatMolSettingsDialog(QDialog):

        def __init__(self, agent, parent=None):
            super().__init__(parent)
            self.agent = agent
            self.setWindowTitle("ChatMol Settings")
            self.setMinimumWidth(460)
            self._build_ui()

        def _build_ui(self):
            layout = QFormLayout(self)

            # Provider selector
            self.provider_combo = QComboBox()
            for key, prov in PROVIDERS.items():
                self.provider_combo.addItem(prov["label"], key)
            cur_prov = self.agent.config.get("provider", "openrouter")
            idx = self.provider_combo.findData(cur_prov)
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
            self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
            layout.addRow("Provider:", self.provider_combo)

            # Base URL (supports API base such as .../v1 or the full endpoint)
            self.base_url_edit = QLineEdit()
            self.base_url_edit.setPlaceholderText("https://your-newapi.example/v1")
            self.base_url_hint = QLabel(
                "Use an OpenAI-compatible API base URL (.../v1) or full .../chat/completions endpoint"
            )
            self.base_url_hint.setWordWrap(True)
            self.base_url_hint.setStyleSheet("color: grey; font-size: 10px;")
            base_layout = QVBoxLayout()
            base_layout.setSpacing(2)
            base_layout.addWidget(self.base_url_edit)
            base_layout.addWidget(self.base_url_hint)
            layout.addRow("Base URL:", base_layout)

            # API key
            self.api_key_edit = QLineEdit()
            self.api_key_edit.setEchoMode(QLineEdit.Password)
            self.api_key_edit.setText(
                self.agent.config.get("api_keys", {}).get(cur_prov, "")
            )
            self.env_hint = QLabel()
            self.env_hint.setStyleSheet("color: grey; font-size: 10px;")
            key_layout = QVBoxLayout()
            key_layout.setSpacing(2)
            key_layout.addWidget(self.api_key_edit)
            key_layout.addWidget(self.env_hint)
            layout.addRow("API Key:", key_layout)

            # Text model
            self.text_model_combo = QComboBox()
            self.text_model_combo.setEditable(True)
            layout.addRow("Text Model:", self.text_model_combo)

            # Vision model
            self.vision_model_combo = QComboBox()
            self.vision_model_combo.setEditable(True)
            layout.addRow("Vision Model:", self.vision_model_combo)

            self.refresh_models_btn = QPushButton("Refresh Models (/v1/models)")
            self.refresh_models_btn.clicked.connect(self._refresh_models)
            layout.addRow("Available Models:", self.refresh_models_btn)

            # Temperature
            self.temp_spin = QDoubleSpinBox()
            self.temp_spin.setRange(0.0, 2.0)
            self.temp_spin.setSingleStep(0.05)
            self.temp_spin.setValue(self.agent.config.get("temperature", 0.01))
            layout.addRow("Temperature:", self.temp_spin)

            # Max tokens
            self.max_tokens_spin = QSpinBox()
            self.max_tokens_spin.setRange(256, 16384)
            self.max_tokens_spin.setSingleStep(256)
            self.max_tokens_spin.setValue(self.agent.config.get("max_tokens", 4096))
            layout.addRow("Max Tokens:", self.max_tokens_spin)

            # Max iterations
            self.max_iterations_spin = QSpinBox()
            self.max_iterations_spin.setRange(1, 500)
            self.max_iterations_spin.setSingleStep(1)
            self.max_iterations_spin.setValue(
                self.agent.config.get("max_iterations", 50)
            )
            layout.addRow("Max Iterations:", self.max_iterations_spin)

            # Max tool calls per request
            self.max_tool_calls_spin = QSpinBox()
            self.max_tool_calls_spin.setRange(8, 200)
            self.max_tool_calls_spin.setSingleStep(1)
            self.max_tool_calls_spin.setValue(
                self.agent.config.get("max_tool_calls", 30)
            )
            layout.addRow("Max Tool Calls:", self.max_tool_calls_spin)

            # Buttons
            btn_layout = QHBoxLayout()
            save_btn = QPushButton("Save")
            save_btn.clicked.connect(self._on_save)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(self.reject)
            btn_layout.addStretch()
            btn_layout.addWidget(save_btn)
            btn_layout.addWidget(cancel_btn)
            layout.addRow(btn_layout)

            self._populate_models(cur_prov)

        def _on_provider_changed(self, _index):
            prov_key = self.provider_combo.currentData()
            prov = PROVIDERS.get(prov_key, PROVIDERS["openrouter"])
            stored_key = self.agent.config.get("api_keys", {}).get(prov_key, "")
            self.api_key_edit.setText(stored_key)
            stored_base = self.agent.config.get("base_urls", {}).get(prov_key, "")
            self.base_url_edit.setText(stored_base or prov.get("base_url", ""))
            self._populate_models(prov_key)

        def _populate_models(self, prov_key):
            prov = PROVIDERS.get(prov_key, PROVIDERS["openrouter"])
            stored_base = self.agent.config.get("base_urls", {}).get(prov_key, "")
            self.base_url_edit.setText(stored_base or prov.get("base_url", ""))

            env_var = prov.get("env_var", "")
            has_env = bool(os.getenv(env_var, ""))
            if has_env:
                self.env_hint.setText(f"Using env var {env_var} (overrides this field)")
            else:
                self.env_hint.setText(f"Or set env var: {env_var}")

            # Text models
            self.text_model_combo.clear()
            for model_id, label in prov["models"]:
                self.text_model_combo.addItem(f"{label}  ({model_id})", model_id)
            saved = self.agent.config.get("text_model", "")
            restored = False
            for i in range(self.text_model_combo.count()):
                if self.text_model_combo.itemData(i) == saved:
                    self.text_model_combo.setCurrentIndex(i)
                    restored = True
                    break
            if not restored and saved:
                self.text_model_combo.setEditText(saved)

            # Vision models
            self.vision_model_combo.clear()
            self.vision_model_combo.addItem("(none)", "")
            for model_id, label in prov.get("vision_models", []):
                self.vision_model_combo.addItem(f"{label}  ({model_id})", model_id)
            saved_v = self.agent.config.get("vision_model", "")
            restored_v = False
            for i in range(self.vision_model_combo.count()):
                if self.vision_model_combo.itemData(i) == saved_v:
                    self.vision_model_combo.setCurrentIndex(i)
                    restored_v = True
                    break
            if not restored_v and saved_v:
                self.vision_model_combo.setEditText(saved_v)

        def _refresh_models(self):
            prov_key = self.provider_combo.currentData()
            prov = PROVIDERS.get(prov_key, PROVIDERS["openrouter"])
            api_key = self.api_key_edit.text().strip() or os.getenv(
                prov.get("env_var", ""), ""
            )
            base_url = self.base_url_edit.text().strip() or prov.get("base_url", "")
            client = LLMClient(prov_key, api_key, base_url)
            try:
                models = client.list_models()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "ChatMol - Model Discovery", f"Could not fetch /models:\n{exc}"
                )
                return
            if not models:
                QtWidgets.QMessageBox.information(
                    self,
                    "ChatMol - Model Discovery",
                    "The endpoint responded successfully but returned no model IDs.",
                )
                return

            current_text = self.text_model_combo.currentText().strip()
            current_vision = self.vision_model_combo.currentText().strip()

            self.text_model_combo.clear()
            for model_id in models:
                self.text_model_combo.addItem(model_id, model_id)
            if current_text:
                idx = self.text_model_combo.findData(current_text)
                if idx >= 0:
                    self.text_model_combo.setCurrentIndex(idx)
                else:
                    self.text_model_combo.setEditText(current_text)

            self.vision_model_combo.clear()
            self.vision_model_combo.addItem("(none)", "")
            for model_id in models:
                self.vision_model_combo.addItem(model_id, model_id)
            if current_vision and current_vision != "(none)":
                idx = self.vision_model_combo.findData(current_vision)
                if idx >= 0:
                    self.vision_model_combo.setCurrentIndex(idx)
                else:
                    self.vision_model_combo.setEditText(current_vision)

            QtWidgets.QMessageBox.information(
                self,
                "ChatMol - Model Discovery",
                f"Loaded {len(models)} models from the configured endpoint.",
            )

        def _on_save(self):
            prov_key = self.provider_combo.currentData()
            self.agent.config["provider"] = prov_key

            if not isinstance(self.agent.config.get("api_keys"), dict):
                self.agent.config["api_keys"] = {}
            self.agent.config["api_keys"][prov_key] = self.api_key_edit.text().strip()

            if not isinstance(self.agent.config.get("base_urls"), dict):
                self.agent.config["base_urls"] = {}
            entered_base = self.base_url_edit.text().strip().rstrip("/")
            default_base = PROVIDERS.get(prov_key, {}).get("base_url", "").rstrip("/")
            if entered_base and (prov_key == "newapi" or entered_base != default_base):
                self.agent.config["base_urls"][prov_key] = entered_base
            else:
                self.agent.config["base_urls"].pop(prov_key, None)

            text_idx = self.text_model_combo.currentIndex()
            text_data = self.text_model_combo.itemData(text_idx)
            self.agent.config["text_model"] = (
                text_data if text_data else self.text_model_combo.currentText().strip()
            )

            vis_idx = self.vision_model_combo.currentIndex()
            vis_data = self.vision_model_combo.itemData(vis_idx)
            self.agent.config["vision_model"] = (
                vis_data
                if vis_data is not None
                else self.vision_model_combo.currentText().strip()
            )

            self.agent.config["temperature"] = self.temp_spin.value()
            self.agent.config["max_tokens"] = self.max_tokens_spin.value()
            self.agent.config["max_iterations"] = self.max_iterations_spin.value()
            self.agent.config["max_tool_calls"] = self.max_tool_calls_spin.value()
            self.agent._save_config()
            self.agent._reinit_client()
            self.accept()

    # -- ChatMolChatBar -----------------------------------------------------

    class ChatMolChatBar(QDockWidget):

        def __init__(self, agent, parent=None):
            super().__init__("ChatMol", parent)
            self.agent = agent
            self._worker = None
            self._build_ui()

        def _build_ui(self):
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(4, 4, 4, 4)
            vbox.setSpacing(4)

            # Top bar
            top_bar = QHBoxLayout()
            top_bar.addWidget(QLabel("<b>ChatMol</b>"))
            top_bar.addStretch()
            self.mode_combo = QComboBox()
            self.mode_combo.addItem("Auto", "auto")
            self.mode_combo.addItem("Confirm", "confirm")
            self.mode_combo.addItem("Dry Run", "dry_run")
            current_mode = self.agent.config.get("execution_mode", "auto")
            mode_idx = self.mode_combo.findData(current_mode)
            if mode_idx >= 0:
                self.mode_combo.setCurrentIndex(mode_idx)
            self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
            self.mode_combo.setToolTip("AI execution mode")
            top_bar.addWidget(self.mode_combo)

            undo_btn = QPushButton("Undo")
            undo_btn.setFixedWidth(50)
            undo_btn.clicked.connect(self._undo_last_action)
            top_bar.addWidget(undo_btn)

            settings_btn = QPushButton("Settings")
            settings_btn.setFixedWidth(70)
            settings_btn.clicked.connect(self._open_settings)
            top_bar.addWidget(settings_btn)
            about_btn = QPushButton("About")
            about_btn.setFixedWidth(55)
            about_btn.clicked.connect(chatmol_about)
            top_bar.addWidget(about_btn)
            clear_btn = QPushButton("Clear")
            clear_btn.setFixedWidth(50)
            clear_btn.clicked.connect(self._clear_chat)
            top_bar.addWidget(clear_btn)
            vbox.addLayout(top_bar)

            # Execution trace panel
            self.trace_display = QTextEdit()
            self.trace_display.setReadOnly(True)
            self.trace_display.setMaximumHeight(92)
            self.trace_display.setStyleSheet(
                "font-family: Menlo, Consolas, monospace; font-size:10px; "
                "color:#263238; background:#FAFAFA;"
            )
            vbox.addWidget(self.trace_display)

            # Chat history
            self.chat_display = QTextEdit()
            self.chat_display.setReadOnly(True)
            self.chat_display.setMinimumHeight(120)
            vbox.addWidget(self.chat_display)

            # Thinking indicator
            self.thinking = ThinkingIndicator()
            vbox.addWidget(self.thinking)

            # Input bar
            input_bar = QHBoxLayout()
            self.input_edit = QLineEdit()
            self.input_edit.setPlaceholderText("Type a message...")
            self.input_edit.returnPressed.connect(self._on_send)
            input_bar.addWidget(self.input_edit)
            self.send_btn = QPushButton("Send")
            self.send_btn.setFixedWidth(50)
            self.send_btn.clicked.connect(self._on_send)
            input_bar.addWidget(self.send_btn)
            self.model_label = QLabel()
            self.model_label.setStyleSheet("color: grey; font-size: 10px;")
            self._update_model_label()
            input_bar.addWidget(self.model_label)
            vbox.addLayout(input_bar)

            self.setWidget(container)

        def _update_model_label(self):
            prov_name = self.agent.config.get("provider", "openrouter")
            prov_label = PROVIDERS.get(prov_name, {}).get("label", prov_name)
            model = self.agent.config.get("text_model", "")
            short = model.split("/")[-1] if "/" in model else model
            self.model_label.setText(f"{short} ({prov_label})")

        def _on_mode_changed(self, _index):
            mode = self.mode_combo.currentData() or "auto"
            self.agent.config["execution_mode"] = mode
            self.agent._save_config()
            self._append_trace(f"[mode] {mode}")

        def _undo_last_action(self):
            result = undo_last_action()
            if result.get("ok"):
                self._append_trace(
                    f"[undo] restored {result.get('restored', 'previous state')}"
                )
                self._append_html("<b>ChatMol:</b> Restored the previous AI snapshot.")
            else:
                self._append_trace(f"[undo] {result.get('error', 'Undo failed')}")

        def _open_settings(self):
            dlg = ChatMolSettingsDialog(self.agent, self)
            if dlg.exec_():
                self._update_model_label()

        def _clear_chat(self):
            self.chat_display.clear()
            self.trace_display.clear()
            self.agent.reset_conversation()

        def _append_html(self, html):
            self.chat_display.append(html)
            sb = self.chat_display.verticalScrollBar()
            sb.setValue(sb.maximum())

        def _append_trace(self, text):
            self.trace_display.append(_escape_html(text))
            sb = self.trace_display.verticalScrollBar()
            sb.setValue(sb.maximum())

        def _on_send(self):
            text = self.input_edit.text().strip()
            if not text:
                return

            if not self.agent._resolve_api_key():
                self._append_html(
                    '<span style="color:red;">No API key set. '
                    "Open Settings to configure your API key.</span>"
                )
                return

            self.input_edit.clear()
            self.input_edit.setEnabled(False)
            self.send_btn.setEnabled(False)
            self.trace_display.clear()

            self._append_html(
                f'<b style="color:#2962FF;">You:</b> {_escape_html(text)}'
            )

            self.thinking.start("Thinking")

            self._worker = AgentWorker(self.agent, text)
            self._worker.request_tool_execution.connect(
                self._execute_tool_on_main_thread
            )
            self._worker.phase_changed.connect(self._on_phase_changed)
            self._worker.tool_executed.connect(self._on_tool_executed)
            self._worker.finished.connect(self._on_agent_finished)
            self._worker.error.connect(self._on_agent_error)
            self._worker.start()

        def _on_phase_changed(self, phase_text):
            self.thinking.set_phase(phase_text)
            if "Retry" in phase_text:
                self._append_trace(f"[phase] {phase_text}")

        def _on_tool_executed(self, tool_name, result_preview):
            self._append_trace(f"[tool:{tool_name}] {result_preview}")

        def _execute_tool_on_main_thread(self, _call_id, tool_name, args_json):
            try:
                arguments = json.loads(args_json)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            self._append_trace(f"[call:{tool_name}] {args_json[:180]}")
            try:
                mode = self.agent.config.get("execution_mode", "auto")
                if is_mutating_tool(tool_name) and mode == "dry_run":
                    result = preview_tool_call(tool_name, arguments)
                elif is_mutating_tool(tool_name) and mode == "confirm":
                    details = json.dumps(arguments or {}, ensure_ascii=False, indent=2)
                    answer = QtWidgets.QMessageBox.question(
                        self,
                        "Confirm ChatMol action",
                        f"Allow ChatMol to run {tool_name}?\n\n{details[:1800]}",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if answer != QtWidgets.QMessageBox.Yes:
                        result = json.dumps(
                            {
                                "ok": False,
                                "tool": tool_name,
                                "denied": True,
                                "executed": False,
                                "message": "User denied this action. Do not assume state changed.",
                            },
                            ensure_ascii=False,
                        )
                    else:
                        result = execute_tool(
                            tool_name,
                            arguments,
                            client=self.agent.client,
                            vision_model=self.agent.config.get("vision_model", ""),
                        )
                else:
                    result = execute_tool(
                        tool_name,
                        arguments,
                        client=self.agent.client,
                        vision_model=self.agent.config.get("vision_model", ""),
                    )
            except Exception as exc:
                result = f"Tool error: {exc}"
            if self._worker is not None:
                self._worker.deliver_tool_result(result)

        def _on_agent_finished(self, response):
            self.thinking.stop()
            self._append_html(f"<b>ChatMol:</b> {_escape_html(response)}")
            self.input_edit.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_edit.setFocus()

        def _on_agent_error(self, error_msg):
            self.thinking.stop()
            prov = self.agent.config.get("provider", "?")
            model = self.agent.config.get("text_model", "?")
            self._append_trace(f"[error] {error_msg}")
            self._append_html(
                f'<span style="color:red;"><b>Error</b> '
                f"[{_escape_html(prov)}/{_escape_html(model)}]: "
                f"{_escape_html(error_msg)}</span>"
            )
            self.input_edit.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.input_edit.setFocus()


def _escape_html(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# ---------------------------------------------------------------------------
# 6. Module-level initialisation
# ---------------------------------------------------------------------------

_agent = ChatMolAgent()

cmd.extend("chat", _agent.chat)
cmd.extend("set_provider", _agent.set_provider)
cmd.extend("set_base_url", _agent.set_base_url)
cmd.extend("set_api_key", _agent.set_api_key)
cmd.extend("set_model", _agent.set_model)
cmd.extend("set_vision_model", _agent.set_vision_model)
cmd.extend("reset_conversation", _agent.reset_conversation)
cmd.extend("save_conversation", _agent.save_conversation)
cmd.extend("load_conversation", _agent.load_conversation)
cmd.extend("chatmol_config", _agent.show_config)


def chatmol_settings():
    """Open the ChatMol settings dialog (requires Qt)."""
    if not _HAS_QT:
        print(
            "Qt not available. Configure via commands: "
            "set_provider, set_base_url, set_api_key, set_model, set_vision_model"
        )
        return
    app = QApplication.instance()
    if app is None:
        print("No Qt application running.")
        return
    dlg = ChatMolSettingsDialog(_agent)
    dlg.exec_()


def chatmol_gui():
    """Open the ChatMol chat bar (requires Qt)."""
    _init_gui()


def chatmol_about():
    """Show basic plugin information."""
    message = (
        f"ChatMol AI v{__version__}\n\n"
        "Agentic PyMOL assistant with OpenAI-compatible providers, "
        "including custom NewAPI endpoints.\n\n"
        "Commands: chat, set_provider, set_base_url, set_api_key, "
        "set_model, set_vision_model, chatmol_config"
    )
    if _HAS_QT:
        try:
            from pymol.Qt.QtWidgets import QMessageBox
            QMessageBox.information(None, "ChatMol AI", message)
            return
        except Exception:
            pass
    print(message)


def __init_plugin__(app=None):
    """Standard PyMOL plugin entry point used by Plugin Manager."""
    try:
        from pymol.plugins import addmenuitemqt
        addmenuitemqt("ChatMol AI", chatmol_gui)
        print("ChatMol AI registered in the PyMOL Plugin menu.")
    except Exception as exc:
        print(f"ChatMol AI: failed to register Plugin menu item: {exc}")


def chatmol_undo():
    """PyMOL command: chatmol_undo"""
    result = undo_last_action()
    print(json.dumps(result, ensure_ascii=False))


def chatmol_models():
    """PyMOL command: chatmol_models"""
    try:
        models = _agent.fetch_available_models()
        print("\n".join(models) if models else "No models returned.")
    except Exception as exc:
        print(f"Model discovery failed: {exc}")


cmd.extend("chatmol_settings", chatmol_settings)
cmd.extend("chatmol_gui", chatmol_gui)
cmd.extend("chatmol_about", chatmol_about)
cmd.extend("chatmol_undo", chatmol_undo)
cmd.extend("chatmol_models", chatmol_models)


_chatbar = None


def _init_gui():
    """Find PyMOL's main window and dock the ChatMol chat bar."""
    global _chatbar
    if not _HAS_QT:
        print(
            "Qt not available — ChatMol GUI disabled. "
            "Use command-line: chat <message>"
        )
        return
    if _chatbar is not None:
        _chatbar.show()
        _chatbar.raise_()
        return

    app = QApplication.instance()
    if app is None:
        return

    main_win = None
    for w in app.topLevelWidgets():
        if w.inherits("QMainWindow") and w.isVisible():
            main_win = w
            break

    if main_win is None:
        print("ChatMol: could not find PyMOL main window for docking.")
        return

    _chatbar = ChatMolChatBar(_agent, main_win)
    main_win.addDockWidget(Qt.BottomDockWidgetArea, _chatbar)
    print("ChatMol chat bar loaded.")


print(
    "ChatMol plugin loaded. Commands: chat, set_provider, set_base_url, set_api_key, "
    "set_model, set_vision_model, reset_conversation, "
    "chatmol_config, chatmol_settings, chatmol_gui, chatmol_about, "
    "chatmol_undo, chatmol_models"
)
