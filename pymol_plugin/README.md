# ChatMol PyMOL Plugin

An LLM-powered agentic plugin for PyMOL that translates natural language into molecular visualizations.

## Versions

| Directory | Description                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `v1/`     | Original plugin — direct LLM-to-command translation. Supports OpenAI, Anthropic, DeepSeek, Ollama, and the free ChatMol service.                 |
| `v2/`     | Agentic plugin — tool-calling loop with session inspection, vision feedback, and Qt5 GUI. Uses OpenRouter and other OpenAI-compatible providers. |

## v2 — Agentic Plugin

### Installation

**Recommended: install the ZIP plugin.**

1. Download `chatmol-ai-vX.Y.Z.zip` from a GitHub Release or the `chatmol-ai-pymol-plugin` Actions artifact.
2. Open PyMOL.
3. Go to **Plugin → Plugin Manager → Install New Plugin**.
4. Select the ZIP.
5. Open **Plugin → ChatMol AI**.

The ZIP contains the multi-file `chatmol_ai/` package with a standard `__init_plugin__` entry point.

For a full repository clone, the legacy launcher remains available:

```python
run /path/to/ChatMol/pymol_plugin/v2/chatmol.py
```

To build the ZIP yourself:

```bash
python scripts/build_plugin_zip.py
```

### Supported Providers

v2 routes through OpenAI-compatible APIs. Configure via `set_provider`:

| Provider        | Models                                | API Key Env Var      |
| --------------- | ------------------------------------- | -------------------- |
| OpenRouter      | GPT-4o, GPT-5.2, Gemini 3 Flash, etc. | `OPENROUTER_API_KEY` |
| DeepSeek        | DeepSeek V3, DeepSeek R1              | `DEEPSEEK_API_KEY`   |
| Kimi (Moonshot) | Kimi K2.5                             | `MOONSHOT_API_KEY`   |
| GLM (Zhipu)     | GLM-5                                 | `GLM_API_KEY`        |
| Custom / NewAPI  | Any compatible model ID               | `CHATMOL_CUSTOM_API_KEY` |

### Custom / NewAPI / OpenAI-compatible endpoint

Select the `newapi` provider when using NewAPI, LiteLLM, a self-hosted OpenAI-compatible gateway, or another compatible proxy. The Base URL can be either an API base such as `https://api.example.com/v1` or the full `.../chat/completions` endpoint.

Using the Qt settings dialog:

1. Provider: **Custom / NewAPI**
2. Base URL: for example `https://api.example.com/v1`
3. API Key: your gateway key
4. Text Model: the exact model ID exposed by your gateway
5. Click **Refresh Models (/v1/models)** to load model IDs exposed by your NewAPI endpoint.
6. Choose a Text Model and, optionally, a multimodal Vision Model for `capture_viewport` visual QA.

Equivalent PyMOL commands:

```pymol
set_provider newapi
set_base_url https://api.example.com/v1
set_api_key sk-xxxx
set_model your-text-model-id
set_vision_model your-vision-model-id
chat inspect the current structure and highlight the binding interface
```

The plugin accepts either a Base URL or the full chat-completions endpoint. Text and vision requests use the same configured endpoint.

### Quick Start

```pymol
# Set your API key (saved to ~/.PyMOL/chatmol_config.json)
set_api_key sk-or-xxxx

# Chat with the agent
chat fetch 1ubq and show as cartoon

# Multi-step request
chat fetch 3wzm, show enzyme-substrate interactions in chain A with publication quality
```

![](./demo.png)

### Commands

| Command                | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `chat <message>`       | Send a message to the agent                       |
| `set_provider <name>`  | Switch provider (openrouter, deepseek, kimi, glm, newapi) |
| `set_base_url <url>`    | Set OpenAI-compatible Base URL for current provider |
| `set_api_key <key>`    | Set API key for the current provider              |
| `set_model <model>`    | Set the text model                                |
| `set_vision_model <m>` | Set the vision model for visual QA                |
| `reset_conversation`   | Clear conversation history                        |
| `save_conversation`    | Save conversation to JSON                         |
| `load_conversation`    | Load conversation from JSON                       |
| `chatmol_config`       | Show current configuration                        |
| `chatmol_settings`     | Open the Qt settings dialog                       |
| `chatmol_gui`          | Open the Qt chat bar                              |
| `chatmol_models`       | Fetch and print model IDs from the current `/models` endpoint |
| `chatmol_undo`         | Restore the most recent AI-created PyMOL snapshot |

### Architecture

The v2 implementation is now split into a multi-file package:

```text
pymol_plugin/chatmol_ai/
├── __init__.py
├── core.py
├── providers.py
├── tools.py
└── version.py
```

The agent prefers structured tools for common operations and keeps `run_pymol_commands` as a fallback.

Structured tools include:

- `select_residues`
- `color_selection`
- `show_representation`
- `measure_distance`
- `align_structures`
- `get_sequence`
- `get_contacts`
- `inspect_session`
- `render`
- `capture_viewport`

A safety blocklist still protects the raw command fallback.

### Execution safety and Undo

The ChatMol panel provides three execution modes:

- **Auto** — run approved PyMOL tools immediately.
- **Confirm** — ask before each scene-mutating tool call.
- **Dry Run** — show the planned tool call without changing PyMOL.

Before mutating AI tool calls, ChatMol stores an in-memory PyMOL session snapshot. Use the **Undo** button or `chatmol_undo` to restore the latest snapshot.

### Configuration

Settings are persisted to `~/.PyMOL/chatmol_config.json`:

- `provider` — API provider name
- `api_keys` — per-provider API keys
- `base_urls` — optional per-provider Base URL overrides (required for `newapi`)
- `text_model` — model for chat completions
- `vision_model` — model for visual QA (capture_viewport)
- `execution_mode` — `auto`, `confirm`, or `dry_run`
- `temperature`, `max_tokens`, `max_iterations`, `max_tool_calls`

## v1 — Original Plugin

### Installation

```python
load https://chatmol.com/pymol_plugins/chatmol-latest.py
```

### Supported Providers

| Provider  | Models               | GPU Required | API Key Required | Notes               |
| --------- | -------------------- | ------------ | ---------------- | ------------------- |
| OpenAI    | GPT Models           | No           | Yes              | Commercial API      |
| Anthropic | Claude Models        | No           | Yes              | Commercial API      |
| DeepSeek  | DeepSeek Models      | No           | Yes              | Commercial API      |
| Ollama    | LLaMA, Mixtral, etc. | Yes          | No               | Self-hosted models  |
| ChatMol   | ChatMol Model        | No           | No               | Free hosted service |

### Quick Start

```pymol
# Free service, no API key needed
chatlite show me a protein

# With an API key
set_api_key openai, sk-proj-xxxx
chat show me a protein
```

For self-hosted models via Ollama:
```pymol
update_model phi-4@ollama
```
