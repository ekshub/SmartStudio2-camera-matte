import time
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.audio.live_subtitles import LiveSubtitleEngine, _to_mono_16k
from src.utils.config import SubtitleConfig, load_config


class _MockModel:
    def transcribe(self, audio, **kwargs):
        assert audio.dtype == np.float32
        assert kwargs["beam_size"] == 1
        return iter([SimpleNamespace(text=" 你好世界 ")]), SimpleNamespace(
            language="zh", language_probability=0.987
        )


class _RecordingModel:
    def __init__(self):
        self.calls = []
        self.called = threading.Event()

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio.copy(), kwargs))
        if len(self.calls) >= 2:
            self.called.set()
        return iter([SimpleNamespace(text=" lyrics ", start=0.0, end=len(audio) / 16000)]), SimpleNamespace(
            language="zh", language_probability=0.9
        )


class _RealtimeMockSource:
    def __init__(self):
        self.closed = False

    def read(self, frames):
        time.sleep(0.01)
        return np.ones(frames, dtype=np.float32)

    def close(self):
        self.closed = True


def test_default_subtitle_config_is_private_and_off():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.subtitles.enabled is False
    assert cfg.subtitles.model == "models/faster-whisper-large-v3-turbo"
    assert cfg.subtitles.language == "auto"
    assert cfg.subtitles.device == "auto"
    assert cfg.subtitles.audio_source == "auto"
    assert cfg.subtitles.microphone_device == "default"
    assert cfg.subtitles.capture_seconds == 0.5
    assert cfg.subtitles.window_seconds == 10.0
    assert cfg.subtitles.minimum_seconds == 1.0
    assert cfg.subtitles.display_seconds == 3.5
    assert cfg.subtitles.update_seconds == 1.0


def test_disabled_engine_does_not_load_model(tmp_path):
    loaded = []
    engine = LiveSubtitleEngine(
        SubtitleConfig(enabled=False),
        tmp_path,
        model_factory=lambda *args: loaded.append(args),
    )
    try:
        time.sleep(0.05)
        assert engine.status()["status"] == "disabled"
        assert loaded == []
    finally:
        engine.close()


def test_recognition_updates_text_and_detected_language(tmp_path):
    engine = LiveSubtitleEngine(SubtitleConfig(enabled=False), tmp_path)
    try:
        engine._model = _MockModel()
        engine._recognize(np.zeros(16000, dtype=np.float32))
        status = engine.status()
        assert status["text"] == "你好世界"
        assert status["language"] == "zh"
        assert status["language_probability"] == 0.987
        assert status["status"] == "listening"
    finally:
        engine.close()


def test_web_template_contains_unicode_subtitle_overlay():
    html = (Path(__file__).parents[1] / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="subtitle"' in html
    assert 'id="sub_enabled"' in html
    assert 'id="sub_audio_source"' in html
    assert 'id="sub_microphone"' in html
    assert 'id="sub_system_audio"' in html
    assert 'id="refresh_audio"' in html
    assert "音频设备已刷新" in html
    assert "实时字幕" in html


def test_audio_source_can_be_selected_independently(tmp_path):
    engine = LiveSubtitleEngine(SubtitleConfig(enabled=False), tmp_path)
    try:
        engine.set_source("camera", 0)
        engine.configure(audio_source="system", system_audio_device="17")
        status = engine.status()
        assert status["audio_source"] == "system"
        assert status["audio_device"] == "17"
        engine.configure(audio_source="microphone", microphone_device="14")
        status = engine.status()
        assert status["audio_source"] == "microphone"
        assert status["audio_device"] == "14"
    finally:
        engine.close()


def test_native_stereo_audio_is_resampled_to_whisper_rate():
    stereo = np.column_stack((np.arange(4800), np.arange(4800))).astype("<i2")
    audio = _to_mono_16k(stereo.tobytes(), 2, 48000, 16000, 1600)
    assert audio.dtype == np.float32
    assert audio.shape == (1600,)


def test_system_audio_uses_realtime_window_without_speech_vad(tmp_path):
    model = _RecordingModel()
    source = _RealtimeMockSource()
    engine = LiveSubtitleEngine(
        SubtitleConfig(
            enabled=True,
            language="zh",
            audio_source="system",
            capture_seconds=0.5,
            window_seconds=10.0,
            minimum_seconds=1.0,
            update_seconds=1.0,
        ),
        tmp_path,
        model_factory=lambda *_: model,
        source_factory=lambda *_: source,
    )
    try:
        assert model.called.wait(2.5)
        assert model.calls[0][0].shape == (16000,)
        assert model.calls[1][0].shape == (32000,)
        assert model.calls[0][1]["vad_filter"] is False
        assert model.calls[0][1]["without_timestamps"] is False
        status = engine.status()
        assert status["update_interval_ms"] > 0
        assert status["estimated_delay_ms"] >= 1000
    finally:
        engine.close()


def test_microphone_source_is_read_and_transcribed(tmp_path):
    model = _RecordingModel()
    source = _RealtimeMockSource()
    engine = LiveSubtitleEngine(
        SubtitleConfig(
            enabled=True,
            language="zh",
            audio_source="microphone",
            microphone_device="15",
            capture_seconds=0.5,
            window_seconds=10.0,
            minimum_seconds=1.0,
            update_seconds=1.0,
        ),
        tmp_path,
        model_factory=lambda *_: model,
        source_factory=lambda audio_source, _kind, _value, _rate: (
            source if audio_source == "microphone" else (_ for _ in ()).throw(AssertionError())
        ),
    )
    try:
        assert model.called.wait(2.5)
        status = engine.status()
        assert status["audio_source"] == "microphone"
        assert status["audio_device"] == "15"
        assert status["text"] == "lyrics"
    finally:
        engine.close()


def test_timestamp_impossible_hallucination_is_not_displayed(tmp_path):
    engine = LiveSubtitleEngine(SubtitleConfig(enabled=False), tmp_path)
    try:
        engine._model = SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (
                iter([SimpleNamespace(text="any words are valid", start=0.0, end=29.98)]),
                SimpleNamespace(language="zh", language_probability=1.0),
            )
        )
        engine._recognize(np.ones(64000, dtype=np.float32), recent_seconds=3.5)
        assert engine.status()["text"] == ""
    finally:
        engine.close()
