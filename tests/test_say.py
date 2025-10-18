import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import say  # noqa: E402


class DummyResponses:
    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.payload))


class DummySpeechStream:
    def __init__(self):
        self.written_path: Path | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def stream_to_file(self, path: Path) -> None:
        self.written_path = Path(path)
        self.written_path.write_bytes(b"dummy")


class DummyAudioSpeech:
    def __init__(self) -> None:
        self.kwargs = None
        self.output_target = None

    def with_streaming_response(self, **kwargs):
        raise RuntimeError("not expected")  # pragma: no cover

    @property
    def with_streaming_response(self):
        class _Wrapper:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.kwargs = kwargs
                return DummySpeechStream()

        return _Wrapper(self)


class DummyClient:
    def __init__(self, payload: dict[str, str]):
        self.responses = DummyResponses(payload)
        self.audio = SimpleNamespace(speech=DummyAudioSpeech())


def test_normalize_text_success():
    payload = {
        "language_code": "pl",
        "normalized_text": "processed text",
        "notes": ["example note"],
        "segments": ["processed text"],
    }
    client = DummyClient(payload)

    result = say.request_normalization(client, "Sample input", "instructions", temperature=0.1)

    assert result.language_code == "pl"
    assert result.normalized_text == "processed text"
    assert result.notes == ("example note",)
    assert result.segments == ("processed text",)
    assert client.responses.last_kwargs["model"] == say.PREPROCESS_MODEL
    assert client.responses.last_kwargs["input"] == "Sample input"


def test_normalize_text_rejects_bad_json():
    client = DummyClient({})
    client.responses.create = lambda **_: SimpleNamespace(output_text="{bad json")

    with pytest.raises(say.SayError):
        say.request_normalization(client, "text", "instructions", temperature=0.0)


def test_synthesize_speech_streams_to_file(tmp_path: Path):
    client = DummyClient({})
    output = tmp_path / "audio.mp3"

    say.synthesize_speech(client, "sample content", voice="alloy", output_path=output)

    assert output.exists()
    assert output.read_bytes() == b"dummy"
    assert client.audio.speech.kwargs["model"] == say.TTS_MODEL
    assert client.audio.speech.kwargs["input"] == "sample content"


def test_read_input_text_from_file(tmp_path: Path):
    src = tmp_path / "sample.txt"
    src.write_text("file content", encoding="utf-8")
    args = SimpleNamespace(text=None, input_file=src)
    assert say.read_input_text(args) == "file content"


def test_resolve_output_path_creates_directory(tmp_path: Path):
    target = tmp_path / "sub" / "file.mp3"
    path = say.resolve_output_path(target)
    assert path == target
    assert target.parent.exists()


def test_play_audio_invokes_afplay(monkeypatch, tmp_path: Path):
    called = {}

    def fake_run(cmd, check):
        called["cmd"] = cmd
        called["check"] = check

    monkeypatch.setattr(say.subprocess, "run", fake_run)
    sample = tmp_path / "audio.mp3"
    sample.write_bytes(b"data")

    say.play_audio(sample)

    assert called["cmd"][0] == "afplay"
    assert called["cmd"][1] == str(sample)
    assert called["check"] is True


def test_compute_cache_key_changes_with_text():
    prompt = "instructions"
    first = say.compute_cache_key("hello", prompt)
    second = say.compute_cache_key("hello there", prompt)
    assert first != second
    assert first == say.compute_cache_key("hello", prompt)


def test_load_normalization_from_cache_creates_markdown(tmp_path: Path):
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir()
    entry = say.CacheEntryPaths(root=entry_dir)
    metadata = {
        "language_code": "en",
        "normalized_text": "normalized",
        "notes": ["note"],
        "segments": ["normalized"],
    }
    entry.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = say.load_normalization_from_cache(entry)

    assert result is not None
    assert result.normalized_text == "normalized"
    assert result.segments == ("normalized",)
    assert entry.normalized_md_path.exists()


def test_obtain_normalization_uses_cache(tmp_path: Path):
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir()
    entry = say.CacheEntryPaths(root=entry_dir)
    metadata = {
        "language_code": "pl",
        "normalized_text": "tekst",
        "notes": [],
        "segments": ["tekst"],
    }
    entry.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    client = DummyClient({"language_code": "pl", "normalized_text": "ignored", "notes": []})

    result, cached = say.obtain_normalization(
        client=client,
        entry=entry,
        raw_text="raw",
        instructions="instructions",
        temperature=None,
        cache_key="abc",
    )

    assert cached is True
    assert result.normalized_text == "tekst"
    assert client.responses.last_kwargs is None


def test_ensure_audio_file_reuses_existing(tmp_path: Path, monkeypatch):
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir()
    entry = say.CacheEntryPaths(root=entry_dir)
    audio_path = entry.audio_path("Alloy")
    audio_path.write_bytes(b"cached")

    def not_expected(*_args, **_kwargs):
        raise AssertionError("synthesize should not run")

    monkeypatch.setattr(say, "synthesize_speech", not_expected)

    path, cached = say.ensure_audio_file(SimpleNamespace(), "text", "Alloy", entry)

    assert cached is True
    assert path == audio_path


def test_ensure_audio_file_generates_when_missing(tmp_path: Path, monkeypatch):
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir()
    entry = say.CacheEntryPaths(root=entry_dir)

    def fake_synthesize(_client, text, voice, output_path):
        output_path.write_bytes(f"{text}-{voice}".encode("utf-8"))

    monkeypatch.setattr(say, "synthesize_speech", fake_synthesize)

    path, cached = say.ensure_audio_file(SimpleNamespace(), "text", "Alloy", entry)

    assert cached is False
    assert path.exists()
    assert path.read_bytes() == b"text-Alloy"


def test_human_readable_size_formats_bytes():
    assert say.human_readable_size(512) == "512 bytes"
    assert say.human_readable_size(1024) == "1.0 KB"
    assert say.human_readable_size(1536) == "1.5 KB"


def test_load_config_respects_env(monkeypatch, tmp_path: Path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("CACHE_WARNING_THRESHOLD", raising=False)
    monkeypatch.setenv("DEFAULT_VOICE", "nova")

    config = say.load_config()

    assert config.cache_dir == cache_dir
    assert config.default_voice == "nova"
    assert config.cache_warning_threshold == say.CACHE_WARNING_DEFAULT


def test_probe_audio_duration_seconds(monkeypatch):
    class DummyResult:
        stdout = "estimated duration: 3.500000 sec\n"

    monkeypatch.setattr(
        say.subprocess,
        "run",
        lambda *args, **kwargs: DummyResult(),
    )

    duration = say.probe_audio_duration_seconds(Path("dummy.mp3"))
    assert duration == 3.5


def test_probe_audio_duration_handles_missing_tool(monkeypatch):
    def raise_missing(*_args, **_kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(say.subprocess, "run", raise_missing)

    duration = say.probe_audio_duration_seconds(Path("dummy.mp3"))
    assert duration is None


def test_parse_args_no_arguments_prints_examples(capsys):
    with pytest.raises(SystemExit) as exc:
        say.parse_args([])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Examples:" in captured.out


def test_parse_args_missing_input_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        say.parse_args(["--play"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Examples:" in captured.out


def test_parse_args_conflicting_output(capsys):
    with pytest.raises(SystemExit) as exc:
        say.parse_args(["--text", "hello", "--no-output", "-o", "out.mp3"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--no-output cannot be used" in captured.err or "--no-output" in captured.out


def test_parse_args_skip_cache():
    args = say.parse_args(["--text", "hello", "--skip-cache"])
    assert args.skip_cache is True


def test_chunk_text_splits_long_text(monkeypatch):
    encoding = say.get_token_encoding()
    sample = "Zdarzyło się raz. To jest bardzo długi tekst, " * 140
    tokens = len(encoding.encode(sample))
    assert tokens > say.MAX_TOKENS_PER_CHUNK

    chunks = say.chunk_text(sample)

    assert len(chunks) >= 2
    assert all(chunk.endswith(('.', '!', '?')) or len(chunk) < 5 for chunk in chunks[:-1])


def test_merge_audio_files(tmp_path: Path):
    src1 = tmp_path / "a.mp3"
    src2 = tmp_path / "b.mp3"
    src1.write_bytes(b"abc")
    src2.write_bytes(b"def")
    dest = tmp_path / "out.mp3"

    say.merge_audio_files([src1, src2], dest)

    assert dest.read_bytes() == b"abcdef"


def test_play_audio_sequence(monkeypatch, tmp_path: Path):
    calls = []

    def fake_play(path):
        calls.append(str(path))

    monkeypatch.setattr(say, "play_audio", fake_play)
    files = [tmp_path / "seg1.mp3", tmp_path / "seg2.mp3"]
    for file in files:
        file.write_bytes(b"data")

    say.play_audio_sequence(files)

    assert calls == [str(files[0]), str(files[1])]
