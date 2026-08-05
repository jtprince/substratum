# AGENTS.md

## Git policy

Do **not** commit, push, amend, or open PRs unless the user explicitly directs
it. Staging explicit paths, diffing, and reading history are fine.

## Commands

- `uv sync` — install dependencies
- `uv run pytest` — run the test suite
- `uv run ruff check .` and `uv run ruff format .` — lint / format
- `uv run mypy substratum` — type check
- `uv run bass --preset velvet --output out.wav` — render a bass
- `uv run bass gallery` — regenerate the sound gallery
- `uv run python docs/generate_figures.py` — regenerate theory figures

## Conventions

- CLIs are thin: all logic lives in libraries under `substratum/`.
- Keep CLI tests minimal; the DSP/bass libraries carry the coverage.
- Theory docs live in `docs/`; the README stays lean (install + usage).
- All images are reproducible from a script (never hand-drawn binaries).
