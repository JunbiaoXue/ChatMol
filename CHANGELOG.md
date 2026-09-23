# Changelog

## 0.3.2

### Improved
- Add a structured `compose_figure` step that frames the selected molecular focus, hides specified marker clutter, and applies legible figure settings.
- Capture the actual PyMOL viewport dimensions instead of forcing an 800 × 600 preview.
- Make the figure workflow check framing and contrast before final rendering; distinguish membrane marker points from a molecular bilayer.
- Default PNG exports to an opaque background unless transparency is requested.

## 0.3.1

### Fixed
- Let Settings fields grow with the window, with readable Text Model and Vision Model selectors.
- Arrange ChatMol's dock controls in two columns so button labels fit in narrow panels.
- Give the message input and Send button their own row, and wrap the model status below them.

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
