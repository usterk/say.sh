#!/usr/bin/env python3
"""CLI utility that converts text to speech using OpenAI models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List, Set

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import tiktoken

PROMPT_FILE = Path(__file__).with_name("speech_prep_prompt.md")
DEFAULT_VOICE = "alloy"
TTS_MODEL = "gpt-4o-mini-tts"
PREPROCESS_MODEL = "gpt-5-mini"
INPUT_FILENAME = "input.txt"
NORMALIZED_MD_FILENAME = "normalized.md"
METADATA_FILENAME = "metadata.json"
CACHE_WARNING_DEFAULT = 100
AVAILABLE_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]
EXAMPLES = textwrap.dedent(
    """\
    Examples:
      say.py --text \"Hello from CLI\" --play
      say.py --input-file notes.txt --play -o speech.mp3
    """
)
MAX_TOKENS_PER_CHUNK = 1500
TOKEN_ENCODING = None


def get_token_encoding():
    global TOKEN_ENCODING
    if TOKEN_ENCODING is None:
        try:
            TOKEN_ENCODING = tiktoken.encoding_for_model(PREPROCESS_MODEL)
        except KeyError:
            TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return TOKEN_ENCODING


SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?][\"']?(?=\s)|[.!?][\"']?$", re.DOTALL)


def find_chunk_boundary(chunk_text: str) -> Optional[int]:
    boundary = None
    for match in SENTENCE_BOUNDARY_PATTERN.finditer(chunk_text):
        boundary = match.end()
    if boundary is not None and boundary > 0:
        return boundary
    return None


def chunk_text(raw_text: str) -> List[str]:
    text = raw_text.strip()
    if not text:
        return []

    encoding = get_token_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= MAX_TOKENS_PER_CHUNK:
        return [text]

    segments: List[str] = []
    start = 0
    total = len(tokens)

    while start < total:
        end = min(start + MAX_TOKENS_PER_CHUNK, total)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)

        if end < total:
            boundary = find_chunk_boundary(chunk_text)
            if boundary and boundary < len(chunk_text):
                candidate = chunk_text[:boundary].rstrip()
                consumed_tokens = encoding.encode(candidate)
                if consumed_tokens:
                    chunk_tokens = consumed_tokens
                    chunk_text = candidate
                    end = start + len(chunk_tokens)

        segments.append(chunk_text.strip())
        start = end

    return [segment for segment in segments if segment]


def log(message: str, *, stderr: bool = False) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = f"[{timestamp}]: {message}"
    print(output, file=sys.stderr if stderr else sys.stdout)


def log_warning(message: str) -> None:
    log(message, stderr=True)


class SayError(RuntimeError):
    """Raised when the CLI fails to complete its task."""


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    cache_dir: Path
    cache_warning_threshold: int
    default_voice: str


@dataclass(frozen=True)
class NormalizationResult:
    language_code: str
    normalized_text: str
    notes: tuple[str, ...]
    segments: tuple[str, ...]
    segments: tuple[str, ...]


@dataclass(frozen=True)
class CacheEntryPaths:
    root: Path

    @property
    def input_path(self) -> Path:
        return self.root / INPUT_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.root / METADATA_FILENAME

    @property
    def normalized_md_path(self) -> Path:
        return self.root / NORMALIZED_MD_FILENAME

    def audio_path(self, voice: str) -> Path:
        return self.root / f"audio-{sanitize_voice(voice)}.mp3"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    args_iterable = list(argv) if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="say.py",
        description="Process text and generate an audio file using OpenAI Text-to-Speech.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--text", help="Plain text to read aloud.")
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Path to a UTF-8 text file that should be read aloud.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for the MP3 audio file.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the audio immediately after generation (requires `afplay`).",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Name of the TTS voice (overrides env/default). Available: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for text normalization (omit to use the model default).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display extra diagnostics (language, voice, notes, audio size & duration).",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voice presets and exit.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Remove all cached entries and exit.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write a merged output file; keep only cached or temporary segments.",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Process without reading or writing cache entries.",
    )
    if not args_iterable:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args(args_iterable)
    if args.no_output and args.output:
        parser.error("--no-output cannot be used together with --output/-o")
    if not (args.clear_cache or args.list_voices) and not (args.text or args.input_file):
        parser.print_help()
        raise SystemExit(2)
    return args


def load_config() -> AppConfig:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SayError(
            "Missing OPENAI_API_KEY. Add it to the .env file or export it in the environment."
        )
    cache_dir = Path(os.getenv("CACHE_DIR", "~/.say_cache/")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    threshold_raw = os.getenv("CACHE_WARNING_THRESHOLD")
    if threshold_raw is None:
        threshold = CACHE_WARNING_DEFAULT
    else:
        try:
            threshold = int(threshold_raw)
        except ValueError as exc:
            raise SayError("CACHE_WARNING_THRESHOLD must be an integer.") from exc
        if threshold < 0:
            raise SayError("CACHE_WARNING_THRESHOLD must be non-negative.")
    default_voice = os.getenv("DEFAULT_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
    return AppConfig(
        api_key=api_key,
        cache_dir=cache_dir,
        cache_warning_threshold=threshold,
        default_voice=default_voice,
    )


def read_input_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.input_file:
        try:
            return args.input_file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SayError(f"Input file not found: {args.input_file}") from exc
        except OSError as exc:
            raise SayError(f"Failed to read input file: {args.input_file}") from exc
    raise SayError("No input text provided.")


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise SayError(f"Missing preprocessing rules file: {PROMPT_FILE}")
    try:
        return PROMPT_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SayError(f"Failed to read prompt file {PROMPT_FILE}") from exc


def response_text_config() -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "speech_normalization_payload",
            "schema": {
                "type": "object",
                "properties": {
                    "language_code": {
                        "type": "string",
                        "description": "Two-letter ISO-639-1 language code.",
                    },
                    "normalized_text": {
                        "type": "string",
                        "description": "Processed text that is ready for speech synthesis.",
                    },
                    "notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional notes describing unusual elements.",
                    },
                    "segments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of normalized segments in order.",
                    },
                },
                "required": ["language_code", "normalized_text", "notes", "segments"],
                "additionalProperties": False,
            },
        }
    }


def request_normalization(
    client: OpenAI,
    raw_text: str,
    instructions: str,
    temperature: Optional[float],
) -> NormalizationResult:
    request_payload: Dict[str, Any] = {
        "model": PREPROCESS_MODEL,
        "instructions": instructions,
        "input": raw_text,
        "text": response_text_config(),
    }
    if temperature is not None:
        request_payload["temperature"] = temperature

    try:
        response = client.responses.create(**request_payload)
    except OpenAIError as exc:
        raise SayError(f"Text normalization failed: {exc}") from exc

    payload_text = getattr(response, "output_text", None)
    if not payload_text:
        raise SayError("No textual response returned from normalization model.")

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise SayError(f"Normalization response is not valid JSON: {payload_text}") from exc

    try:
        language_code = payload["language_code"]
        normalized_text = payload["normalized_text"]
    except KeyError as exc:
        raise SayError(f"Normalization response is missing field {exc.args[0]}: {payload}") from exc

    notes_raw = payload.get("notes", []) or []
    segments_raw = payload.get("segments")
    notes = tuple(str(note) for note in notes_raw)
    if isinstance(segments_raw, list) and segments_raw:
        segments = tuple(str(seg) for seg in segments_raw)
    else:
        segments = (normalized_text,)
    if not isinstance(language_code, str) or not language_code:
        raise SayError("Field language_code is empty or not a string.")
    if not isinstance(normalized_text, str) or not normalized_text.strip():
        raise SayError("Field normalized_text is empty or invalid.")

    return NormalizationResult(
        language_code=language_code,
        normalized_text=normalized_text,
        notes=notes,
        segments=segments,
    )


def sanitize_voice(voice: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "-" for ch in voice)
    sanitized = "-".join(filter(None, sanitized.split("-")))
    return sanitized or "default"


def human_readable_size(num_bytes: int) -> str:
    units = ["bytes", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "bytes":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} bytes"


def print_voice_catalog(default_voice: str) -> None:
    log("Available voices:")
    known = {voice.lower(): voice for voice in AVAILABLE_VOICES}
    for voice in AVAILABLE_VOICES:
        marker = " (default)" if voice.lower() == default_voice.lower() else ""
        log(f" - {voice}{marker}")
    if default_voice.lower() not in known:
        log(f"Custom default voice from env: {default_voice}")


def probe_audio_duration_seconds(path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            ["afinfo", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None

    for line in result.stdout.splitlines():
        if "estimated duration" in line.lower():
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    return float(parts[2])
                except ValueError:
                    continue
    return None


def merge_audio_files(source_paths: Iterable[Path], destination: Path) -> None:
    sources = [Path(path) for path in source_paths]
    if not sources:
        raise SayError("No audio segments generated.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as dest_file:
        for path in sources:
            with path.open("rb") as src_file:
                shutil.copyfileobj(src_file, dest_file)


def play_audio_sequence(paths: Iterable[Path]) -> None:
    path_list = list(paths)
    total = len(path_list)
    for idx, path in enumerate(path_list, start=1):
        log(f"Playing chunk {idx}/{total}...")
        play_audio(path)
        log(f"Playing chunk {idx}/{total}... done")


def compute_cache_key(raw_text: str, instructions: str) -> str:
    digest = hashlib.md5()
    digest.update(raw_text.encode("utf-8"))
    digest.update(b"\0")
    digest.update(instructions.encode("utf-8"))
    return digest.hexdigest()


def build_cache_entry(cache_dir: Path, cache_key: str) -> CacheEntryPaths:
    root = cache_dir / cache_key
    root.mkdir(parents=True, exist_ok=True)
    return CacheEntryPaths(root=root)


def write_input_snapshot(entry: CacheEntryPaths, raw_text: str) -> None:
    if entry.input_path.exists():
        return
    entry.input_path.write_text(raw_text, encoding="utf-8")


def load_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_normalized_markdown(result: NormalizationResult) -> str:
    lines = [
        "# Normalized Text",
        "",
        f"Language: {result.language_code}",
    ]
    if result.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in result.notes)
    else:
        lines.append("Notes: none")
    lines.extend(
        [
            "",
            "```",
            result.normalized_text,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def ensure_normalized_markdown(entry: CacheEntryPaths, result: NormalizationResult) -> None:
    if entry.normalized_md_path.exists():
        return
    entry.normalized_md_path.write_text(format_normalized_markdown(result), encoding="utf-8")


def load_normalization_from_cache(entry: CacheEntryPaths) -> Optional[NormalizationResult]:
    metadata = load_metadata(entry.metadata_path)
    if not metadata:
        return None
    language_code = metadata.get("language_code")
    normalized_text = metadata.get("normalized_text")
    notes_raw = metadata.get("notes", [])
    segments_raw = metadata.get("segments", [])
    if not isinstance(language_code, str) or not language_code:
        return None
    if not isinstance(normalized_text, str) or not normalized_text.strip():
        return None
    if not isinstance(notes_raw, list):
        notes_raw = []
    if not isinstance(segments_raw, list):
        segments_raw = []
    if not segments_raw:
        segments_raw = [normalized_text]
    notes: Tuple[str, ...] = tuple(str(note) for note in notes_raw)
    segments: Tuple[str, ...] = tuple(str(seg) for seg in segments_raw)
    result = NormalizationResult(
        language_code=language_code,
        normalized_text=normalized_text,
        notes=notes,
        segments=segments,
    )
    ensure_normalized_markdown(entry, result)
    return result


def persist_normalization(
    entry: CacheEntryPaths,
    raw_text: str,
    cache_key: str,
    result: NormalizationResult,
) -> None:
    write_input_snapshot(entry, raw_text)
    metadata = {
        "language_code": result.language_code,
        "normalized_text": result.normalized_text,
        "notes": list(result.notes),
        "segments": list(result.segments),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cache_key": cache_key,
        "schema_version": 1,
    }
    save_metadata(entry.metadata_path, metadata)
    ensure_normalized_markdown(entry, result)


def obtain_normalization(
    client: OpenAI,
    entry: CacheEntryPaths,
    raw_text: str,
    instructions: str,
    temperature: Optional[float],
    cache_key: str,
) -> Tuple[NormalizationResult, bool]:
    cached = load_normalization_from_cache(entry)
    if cached:
        return cached, True
    result = request_normalization(client, raw_text, instructions, temperature)
    persist_normalization(entry, raw_text, cache_key, result)
    return result, False


def ensure_audio_file(
    client: OpenAI,
    normalized_text: str,
    voice: str,
    entry: CacheEntryPaths,
) -> Tuple[Path, bool]:
    target = entry.audio_path(voice)
    if target.exists():
        return target, True
    synthesize_speech(client, normalized_text, voice, target)
    return target, False


def count_cache_entries(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(1 for path in cache_dir.iterdir() if path.is_dir())


def warn_if_cache_large(cache_dir: Path, threshold: int) -> None:
    if threshold <= 0:
        return
    entry_count = count_cache_entries(cache_dir)
    if entry_count > threshold:
        log_warning(
            f"Warning: cache contains {entry_count} entries (threshold {threshold}). Consider clearing it with --clear-cache."
        )


def clear_cache(cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def resolve_output_path(arg_path: Optional[Path]) -> Path:
    if arg_path:
        output_path = arg_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"say-{timestamp}.mp3")
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def synthesize_speech(
    client: OpenAI,
    text: str,
    voice: str,
    output_path: Path,
) -> None:
    try:
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=voice,
            input=text,
            response_format="mp3",
        ) as response:
            response.stream_to_file(output_path)
    except OpenAIError as exc:
        raise SayError(f"Audio synthesis failed: {exc}") from exc
    except OSError as exc:
        raise SayError(f"Failed to write audio file: {exc}") from exc


def play_audio(output_path: Path) -> None:
    try:
        subprocess.run(["afplay", str(output_path)], check=True)
    except FileNotFoundError:
        raise SayError("The `afplay` command is unavailable. Skip --play or install the tool.")
    except subprocess.CalledProcessError as exc:
        raise SayError(f"Audio playback failed: {exc}") from exc


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        args = parse_args(argv)
        config = load_config()
        if args.clear_cache:
            clear_cache(config.cache_dir)
            log(f"Cache cleared at: {config.cache_dir}")
            return 0

        if args.list_voices:
            print_voice_catalog(config.default_voice)
            return 0

        warn_if_cache_large(config.cache_dir, config.cache_warning_threshold)

        client = OpenAI(api_key=config.api_key)
        instructions = load_prompt()
        raw_text = read_input_text(args)
        voice = args.voice or config.default_voice

        chunk_texts = chunk_text(raw_text)
        if not chunk_texts:
            raise SayError("Input text is empty after trimming.")

        normalization_results: List[NormalizationResult] = []
        audio_paths: List[Path] = []
        normalization_cache_hits = 0
        audio_cache_hits = 0
        total_size_bytes = 0
        total_duration_seconds = 0.0
        durations_known = True
        use_cache = not args.skip_cache
        languages_detected: Set[str] = set()
        temp_dir: Optional[tempfile.TemporaryDirectory] = None
        if not use_cache:
            temp_dir = tempfile.TemporaryDirectory()

        if args.verbose:
            log(f"Voice: {voice}")
            log(f"Planned chunks: {len(chunk_texts)}")
            log(f"Cache usage: {'skipped' if args.skip_cache else 'enabled'}")

        try:
            for idx, chunk in enumerate(chunk_texts, start=1):
                chunk_total = len(chunk_texts)
                if args.verbose:
                    log(
                        f"Normalizing chunk {idx}/{chunk_total} ({'cache enabled' if use_cache else 'cache skipped'})..."
                    )
                if use_cache:
                    chunk_key = compute_cache_key(chunk, instructions)
                    entry = build_cache_entry(config.cache_dir, chunk_key)
                    write_input_snapshot(entry, chunk)

                    normalization, normalization_cached = obtain_normalization(
                        client=client,
                        entry=entry,
                        raw_text=chunk,
                        instructions=instructions,
                        temperature=args.temperature,
                        cache_key=chunk_key,
                    )
                    if normalization_cached:
                        normalization_cache_hits += 1

                    if args.verbose:
                        log(f"Preparing audio chunk {idx}/{chunk_total} (cache enabled)...")
                    audio_cache_path, audio_cached = ensure_audio_file(
                        client=client,
                        normalized_text=normalization.normalized_text,
                        voice=voice,
                        entry=entry,
                    )
                    if audio_cached:
                        audio_cache_hits += 1
                else:
                    normalization = request_normalization(
                        client=client,
                        raw_text=chunk,
                        instructions=instructions,
                        temperature=args.temperature,
                    )
                    normalization_cached = False
                    chunk_dir = Path(temp_dir.name)
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    audio_cache_path = chunk_dir / f"chunk-{idx:03}.mp3"
                    if args.verbose:
                        log(f"Preparing audio chunk {idx}/{chunk_total} (generating)...")
                    synthesize_speech(client, normalization.normalized_text, voice, audio_cache_path)
                    audio_cached = False

                normalization_results.append(normalization)
                audio_paths.append(audio_cache_path)

                chunk_language = normalization.language_code or "unknown"
                languages_detected.add(chunk_language)

                chunk_size = audio_cache_path.stat().st_size
                total_size_bytes += chunk_size
                chunk_duration = probe_audio_duration_seconds(audio_cache_path)
                if chunk_duration is not None:
                    total_duration_seconds += chunk_duration
                else:
                    durations_known = False

                if args.verbose:
                    norm_status = (
                        "cache hit"
                        if use_cache and normalization_cached
                        else ("cache miss" if use_cache else "cache skipped")
                    )
                    audio_status = (
                        "cache hit"
                        if use_cache and audio_cached
                        else ("cache miss" if use_cache else "generated")
                    )
                    chunk_language = normalization.language_code or "unknown"
                    size_str = human_readable_size(chunk_size)
                    duration_str = (
                        f"{chunk_duration / 60:.2f} min"
                        if chunk_duration is not None
                        else "unknown"
                    )
                    log(
                        f"Chunk {idx}/{chunk_total} (language {chunk_language}): normalization {norm_status}, "
                        f"audio {audio_status}, size {size_str}, duration {duration_str}"
                    )
                    if normalization.notes:
                        for note in normalization.notes:
                            log(f"Chunk {idx} note: {note}")

            if args.no_output or not args.output:
                output_path = None
            else:
                output_path = resolve_output_path(args.output)
                merge_audio_files(audio_paths, output_path)
                log(f"Audio saved to: {output_path}")

            if args.verbose:
                languages_display = ", ".join(sorted(languages_detected)) if languages_detected else "unknown"
                log(f"Languages: {languages_display}")
                log(
                    f"Chunks processed: {len(chunk_texts)} (normalization cache hits: {normalization_cache_hits}, "
                    f"audio cache hits: {audio_cache_hits})"
                )
                size_str = human_readable_size(total_size_bytes)
                log(f"Total audio size: {size_str}")
                if durations_known:
                    log(f"Total audio duration: {total_duration_seconds / 60:.2f} min")
                else:
                    log("Total audio duration: unknown")

            if args.play:
                play_audio_sequence(audio_paths)
                log("Audio playback completed.")

            return 0
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
    except SayError as exc:
        log(f"Error: {exc}", stderr=True)
        return 1
    except KeyboardInterrupt:
        log("Interrupted by user.", stderr=True)
        return 130



if __name__ == "__main__":
    sys.exit(main())
