# Changelog

## 0.3.0

### Added
- OpenAI-compatible `GET /models` discovery from the Settings dialog.
- Auto / Confirm / Dry Run execution modes.
- In-memory PyMOL session snapshots and one-click Undo for AI mutations.
- Structured PyMOL tools for selections, coloring, representations, distances, alignment, sequence retrieval, and contacts.
- Multi-file `chatmol_ai` plugin package.
- Version module and version display in About/configuration.
- Reproducible ZIP builder for PyMOL Plugin Manager.
- GitHub Actions workflow that syntax-checks and builds the ZIP on `main`, and attaches it to tagged releases.

### Compatibility
- `pymol_plugin/v2/chatmol.py` remains as a compatibility launcher for full repository clones.
