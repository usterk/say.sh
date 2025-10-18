# CLI assumptions for `say.py`

- **Text input**: Provide content explicitly via `--text "..."` or by pointing to a file with `--input-file path`. The options are mutually exclusive; omitting both raises an error.
- **Audio output**: Defaults to an MP3 placed in the working directory named `say-YYYYmmdd-HHMMSS.mp3`. The `-o/--output path` flag lets the user choose a destination; directories are created automatically. Use `--no-output` to skip creating a merged file (audio lives only in cache/temp segments).
- **Playback**: The `--play` switch triggers immediate playback through the macOS `afplay` command. No extra Python audio libraries are required.
- **Chunking**: Long inputs are automatically split into ≈1500-token chunks (preferably on sentence boundaries) before normalization and TTS; each chunk is cached separately.
- **Text preprocessing**: The script loads rules from `speech_prep_prompt.md`, calls `gpt-5-mini` with an enforced JSON schema, and consumes the `normalized_text` field.
- **Speech synthesis**: The normalized text is sent to `gpt-4o-mini-tts` using the default `alloy` voice and MP3 format. Returned audio bytes are written to disk and optionally played.
- **Cache control**: By default results are cached; `--skip-cache` forces fresh requests and stores audio only in temporary files.
- **Configuration**: The OpenAI key comes from `.env` (`OPENAI_API_KEY`). Missing credentials produce a helpful error message.
