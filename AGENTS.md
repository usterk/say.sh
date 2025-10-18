# Repository Guidelines

## Project Structure & Module Organization
- `say.py` — core CLI that orchestrates preprocessing, text-to-speech calls, chunking, caching, and playback helpers.
- `speech_prep_prompt.md` — normalization rulebook sent to `gpt-5-mini` before generating speech.
- `tests/` — pytest suite covering chunking, cache helpers, audio utilities, and CLI argument parsing.
- `say.sh` — wrapper that always runs `say.py` inside the repository virtual environment.
- `env.example` — template for `.env` (set `OPENAI_API_KEY`, `CACHE_DIR`, etc.).

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` — create/enter the project venv.
- `pip install -r requirements.txt` — install dependencies (`openai`, `tiktoken`, `pytest`, etc.).
- `./say.sh --text "Hello" --play` — quick smoke test; uses cache and plays audio via `afplay`.
- `.venv/bin/pytest` — run the entire automated test suite; ensure it passes before pushing.

## Coding Style & Naming Conventions
- Python sources use 4-space indentation, descriptive snake_case for functions, UpperCamelCase for dataclasses.
- Keep helper functions (e.g., `log()`, `play_audio_sequence()`) in `say.py` near related logic; resist splitting files unless it simplifies testing.
- Cache artifacts live in `CACHE_DIR/<hash>`; new helpers should respect this pattern and avoid creating flat temp files.

## Testing Guidelines
- Tests rely on `pytest`; place new cases in `tests/test_say.py` unless a separate module warrants its own file.
- Name tests with `test_<behavior>`; prefer deterministic fixtures and monkeypatching over network calls.
- Run `.venv/bin/pytest` locally before every commit; aim to keep runtime under one second by mocking API calls.

## Commit & Pull Request Guidelines
- Follow concise, imperative commit messages (e.g., “Add quick start docs and gitignore”).
- Each PR should describe the why and the how, reference related issues when available, and include verification steps (commands run, test output, sample CLI invocations).
- Screenshots or audio snippets are optional but encouraged when changing output formatting or playback behavior.
