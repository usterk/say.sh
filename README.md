# say.py – Text-to-Speech Helper

`say.py` preprocesses text with `gpt-5-mini`, turns it into speech via `gpt-4o-mini-tts`, and caches every intermediate result so repeated requests are instant.

## Setup
- Create and activate a virtual environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Quick Start
1. Create the virtual environment and install dependencies (see **Setup** above).
2. Copy the sample environment file and edit it with your OpenAI key:
   ```bash
   cp env.example .env
   # open .env and set OPENAI_API_KEY (optionally adjust CACHE_DIR or DEFAULT_VOICE)
   ```
3. Run the wrapper to ensure the script uses the project venv:
   ```bash
   ./say.sh --text "Hello" --play
   ```

## Environment
- Copy `env.example` to `.env` and edit the values:
  - `OPENAI_API_KEY` – required OpenAI key.
  - `CACHE_DIR` – where cache entries are stored (defaults to `~/.say_cache/`).
  - `CACHE_WARNING_THRESHOLD` – optional limit that triggers a warning when the cache grows too large (default `100`).
  - `DEFAULT_VOICE` – default TTS voice if `--voice` is not supplied (`alloy` by default).

## Usage
- Use the convenience wrapper: `./say.sh --text "Hello" --play` (ensures the project venv is used).
- Default behaviour keeps audio in per-chunk cache only. Provide `-o/--output` if you want a single merged MP3.
- Read inline text:
  ```bash
  .venv/bin/python say.py --text "Hello world 2025" --play
  ```
- Use an input file:
  ```bash
  .venv/bin/python say.py --input-file notes.txt -o speech.mp3
  ```
- Override voice and show verbose diagnostics (languages, voice, chunk timeline, size, duration, notes):
  ```bash
  .venv/bin/python say.py --text "Sample" --voice nova -v
  ```
- Large inputs are automatically split into ~1500-token chunks at sentence boundaries so long files are processed safely.
- Inspect supported voice presets:
  ```bash
  .venv/bin/python say.py --list-voices
  ```
- Reuse cached audio automatically whenever the source text and normalization prompt are the same. To remove all cached artifacts:
  ```bash
  .venv/bin/python say.py --clear-cache
  ```

## Cache Structure
- Each cache entry lives in `CACHE_DIR/<md5>`, where the MD5 is computed from the raw input text plus the normalization prompt.
- An entry contains:
  - `input.txt` – original text fragment (one chunk).
  - `normalized.md` – processed text ready for TTS.
  - `metadata.json` – structured metadata (language code, notes, segments, timestamps).
  - `audio-<voice>.mp3` – synthesized audio per voice (voice names are sanitized before use).
- If more than `CACHE_WARNING_THRESHOLD` entries exist, the CLI prints a warning so you can trim the cache.
- Use `--skip-cache` to bypass cache reads/writes and store audio only in a temporary directory for the current run; combine with `--no-output` for quick one-off playback.
- Verbose mode `-v/--verbose` prints timestamped progress (voice/chunk plan, per-chunk normalization and audio generation, notes, playback progress, totals) as soon as data becomes available.

## Tests
- Run `pytest` from the repository root:
  ```bash
  .venv/bin/pytest
  ```
