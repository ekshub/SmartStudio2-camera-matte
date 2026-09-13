from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.utils.config import SubtitleConfig


def _to_mono_16k(raw: bytes, channels: int, source_rate: int, target_rate: int, target_frames: int) -> np.ndarray:
    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)
    usable = samples.size - samples.size % channels
    samples = samples[:usable].reshape(-1, channels).astype(np.float32).mean(axis=1) / 32768.0
    if source_rate == target_rate:
        return samples[:target_frames]
    positions = np.linspace(0, max(0, samples.size - 1), target_frames)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def list_audio_devices() -> dict:
    """Return stable device IDs for the browser without opening any device."""
    result = {"microphones": [], "system_audio": [], "errors": []}
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        wasapi_index = next((i for i, api in enumerate(hostapis) if "WASAPI" in api["name"]), None)
        candidates = [d for d in devices if d["max_input_channels"] > 0]
        if wasapi_index is not None:
            wasapi = [d for d in candidates if d["hostapi"] == wasapi_index]
            candidates = wasapi or candidates
        result["microphones"] = [
            {
                "id": str(d["index"]),
                "name": d["name"],
                "host_api": hostapis[d["hostapi"]]["name"],
                "sample_rate": int(d["default_samplerate"]),
                "channels": int(d["max_input_channels"]),
            }
            for d in candidates
        ]
    except Exception as exc:
        result["errors"].append(f"microphone enumeration failed: {exc}")
    try:
        import pyaudiowpatch as pyaudio

        audio = pyaudio.PyAudio()
        try:
            result["system_audio"] = [
                {
                    "id": str(d["index"]),
                    "name": d["name"].replace(" [Loopback]", ""),
                    "host_api": "Windows WASAPI loopback",
                    "sample_rate": int(d["defaultSampleRate"]),
                    "channels": int(d["maxInputChannels"]),
                }
                for d in audio.get_loopback_device_info_generator()
            ]
        finally:
            audio.terminate()
    except Exception as exc:
        result["errors"].append(f"system audio enumeration failed: {exc}")
    return result


class _FFmpegAudioSource:
    def __init__(self, path: str, sample_rate: int):
        import imageio_ffmpeg

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
            "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "pipe:1",
        ]
        startupinfo = None
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            startupinfo=startupinfo,
        )

    def read(self, frames: int) -> np.ndarray:
        if self.process.stdout is None:
            return np.empty(0, dtype=np.float32)
        expected = frames * 2
        chunks = bytearray()
        while len(chunks) < expected:
            block = self.process.stdout.read(expected - len(chunks))
            if not block:
                break
            chunks.extend(block)
        raw = bytes(chunks)
        if not raw:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.process.stdout is not None:
            self.process.stdout.close()


class _MicrophoneAudioSource:
    def __init__(self, sample_rate: int, device: str = "default"):
        import sounddevice as sd

        selected = None if device == "default" else int(device)
        info = sd.query_devices(selected, "input")
        self.source_rate = int(info["default_samplerate"])
        self.channels = max(1, min(2, int(info["max_input_channels"])))
        self.target_rate = sample_rate
        self.stream = sd.RawInputStream(
            device=selected,
            samplerate=self.source_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=max(1, self.source_rate // 4),
        )
        self.stream.start()

    def read(self, frames: int) -> np.ndarray:
        source_frames = max(1, round(frames * self.source_rate / self.target_rate))
        raw, _overflowed = self.stream.read(source_frames)
        return _to_mono_16k(raw, self.channels, self.source_rate, self.target_rate, frames)

    def close(self):
        self.stream.stop()
        self.stream.close()


class _SystemAudioSource:
    def __init__(self, sample_rate: int, device: str = "default"):
        import pyaudiowpatch as pyaudio

        self.audio = pyaudio.PyAudio()
        info = (
            self.audio.get_default_wasapi_loopback()
            if device == "default"
            else self.audio.get_device_info_by_index(int(device))
        )
        if not info.get("isLoopbackDevice"):
            self.audio.terminate()
            raise ValueError("selected device is not a WASAPI loopback output")
        self.source_rate = int(info["defaultSampleRate"])
        self.channels = max(1, min(2, int(info["maxInputChannels"])))
        self.target_rate = sample_rate
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.source_rate,
            input=True,
            input_device_index=int(info["index"]),
            frames_per_buffer=max(1, self.source_rate // 4),
        )

    def read(self, frames: int) -> np.ndarray:
        source_frames = max(1, round(frames * self.source_rate / self.target_rate))
        raw = self.stream.read(source_frames, exception_on_overflow=False)
        return _to_mono_16k(raw, self.channels, self.source_rate, self.target_rate, frames)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


class LiveSubtitleEngine:
    """Non-blocking subtitles from media, microphones, or system audio."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        config: SubtitleConfig,
        root: str | Path,
        model_factory: Callable | None = None,
        source_factory: Callable | None = None,
    ):
        self.config = config
        self.root = Path(root)
        self._model_factory = model_factory
        self._source_factory = source_factory
        self._model = None
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._running = True
        self._generation = 0
        self._source_type = "video"
        self._source_value = ""
        self._active_source = None
        self._last_voice_at = 0.0
        self._state = {
            "status": "disabled" if not config.enabled else "loading",
            "text": "",
            "language": "",
            "language_probability": 0.0,
            "latency_ms": 0.0,
            "error": "",
            "updated_at": 0.0,
            "device": "",
            "audio_source": self._effective_audio_source(),
            "audio_device": "default",
            "update_interval_ms": 0.0,
        }
        self._last_result_at = 0.0
        self._thread = threading.Thread(target=self._worker, name="smartstudio-subtitles", daemon=True)
        self._thread.start()

    def configure(
        self,
        enabled: bool | None = None,
        language: str | None = None,
        audio_source: str | None = None,
        microphone_device: str | None = None,
        system_audio_device: str | None = None,
    ):
        with self._lock:
            if language is not None:
                language = str(language).strip().lower() or "auto"
                if language not in {"auto", "zh", "en", "ja", "ko", "fr", "de", "es", "ru"}:
                    raise ValueError(f"unsupported subtitle language: {language}")
                self.config.language = language
            if audio_source is not None:
                audio_source = str(audio_source).strip().lower()
                if audio_source not in {"auto", "video", "microphone", "system"}:
                    raise ValueError(f"unsupported audio source: {audio_source}")
                self.config.audio_source = audio_source
            if microphone_device is not None:
                self.config.microphone_device = str(microphone_device)
            if system_audio_device is not None:
                self.config.system_audio_device = str(system_audio_device)
            if enabled is not None:
                self.config.enabled = bool(enabled)
            self._generation += 1
            if not self.config.enabled:
                self._state.update(status="disabled", text="", error="")
            else:
                self._state.update(status="loading" if self._model is None else "listening", error="")
            self._state["audio_source"] = self._effective_audio_source()
            self._state["audio_device"] = self._selected_audio_device()
        self._wake.set()

    def set_source(self, source_type: str, value):
        with self._lock:
            previous_type, previous_value = self._source_type, self._source_value
            previous_audio_source = self._effective_audio_source()
            self._source_type = source_type
            self._source_value = str(value)
            effective = self._effective_audio_source()
            restart = bool(self.config.enabled) and (
                previous_audio_source != effective
                or effective == "video" and (previous_type != source_type or previous_value != str(value))
            )
            self._state["audio_source"] = effective
            if restart:
                self._generation += 1
                self._state.update(text="", language="", language_probability=0.0, updated_at=time.time())
        if restart:
            self._wake.set()

    def _effective_audio_source(self) -> str:
        if self.config.audio_source == "auto":
            return "video" if self._source_type == "video" else "microphone"
        return self.config.audio_source

    def _selected_audio_device(self) -> str:
        source = self._effective_audio_source()
        if source == "microphone":
            return self.config.microphone_device
        if source == "system":
            return self.config.system_audio_device
        return "media"

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": bool(self.config.enabled),
                "status": self._state["status"],
                "text": self._state["text"],
                "language": self._state["language"],
                "language_probability": round(float(self._state["language_probability"]), 3),
                "latency_ms": round(float(self._state["latency_ms"]), 2),
                "error": self._state["error"],
                "updated_at": self._state["updated_at"],
                "device": self._state["device"],
                "audio_source": self._state["audio_source"],
                "audio_device": self._state["audio_device"],
                "update_interval_ms": round(float(self._state["update_interval_ms"]), 2),
                "estimated_delay_ms": round(
                    float(self.config.update_seconds) * 1000 + float(self._state["latency_ms"]), 2
                ),
            }

    def prepare(self):
        """Load the model before media playback is restarted for A/V sync."""
        with self._lock:
            self._state.update(status="loading", error="")
        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return
        model_path = Path(self.config.model)
        if not model_path.is_absolute():
            model_path = self.root / model_path
        device = self.config.device
        compute_type = self.config.compute_type
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        if self._model_factory is not None:
            self._model = self._model_factory(str(model_path), device, compute_type)
            with self._lock:
                self._state["device"] = device
            return
        if not model_path.exists():
            raise FileNotFoundError(f"subtitle model not found: {model_path}")
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute_type,
        )
        with self._lock:
            self._state["device"] = device

    def _open_source(self, audio_source: str, source_type: str, value: str):
        if self._source_factory is not None:
            return self._source_factory(audio_source, source_type, value, self.SAMPLE_RATE)
        if audio_source == "microphone":
            return _MicrophoneAudioSource(self.SAMPLE_RATE, self.config.microphone_device)
        if audio_source == "system":
            return _SystemAudioSource(self.SAMPLE_RATE, self.config.system_audio_device)
        if audio_source != "video":
            raise ValueError(f"unsupported audio source: {audio_source}")
        if source_type != "video":
            raise ValueError("video audio requires a video input source")
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"audio source not found: {value}")
        return _FFmpegAudioSource(value, self.SAMPLE_RATE)

    def _close_active_source(self):
        with self._lock:
            source, self._active_source = self._active_source, None
        if source is not None:
            try:
                source.close()
            except Exception:
                pass

    def _recognize(
        self,
        audio: np.ndarray,
        vad_filter: bool = True,
        recent_seconds: float | None = None,
    ):
        language = None if self.config.language == "auto" else self.config.language
        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
            without_timestamps=recent_seconds is None,
        )
        segments = list(segments)
        audio_seconds = audio.size / self.SAMPLE_RATE
        # Whisper can emit a high-confidence 30-second template for a much
        # shorter non-speech window. Reject timestamp-impossible segments
        # without making assumptions about what the speaker is allowed to say.
        segments = [
            segment
            for segment in segments
            if float(getattr(segment, "start", 0.0)) <= audio_seconds + 0.5
            and float(getattr(segment, "end", audio_seconds)) <= audio_seconds + 0.5
        ]
        if recent_seconds is not None and segments:
            cutoff = max(0.0, audio_seconds - float(recent_seconds))
            recent = [segment for segment in segments if float(getattr(segment, "end", 0.0)) > cutoff]
            if recent:
                segments = recent[-2:]
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        latency_ms = (time.perf_counter() - started) * 1000
        detected_language = getattr(info, "language", language or "") or ""
        probability = getattr(info, "language_probability", 0.0) or 0.0
        now = time.time()
        completed_at = time.perf_counter()
        with self._lock:
            if text:
                self._last_voice_at = now
                self._state.update(text=text, updated_at=now)
            elif now - self._last_voice_at >= float(self.config.hold_seconds):
                self._state.update(text="", updated_at=now)
            self._state.update(
                status="listening",
                language=detected_language,
                language_probability=float(probability),
                latency_ms=latency_ms,
                error="",
                update_interval_ms=(
                    (completed_at - self._last_result_at) * 1000 if self._last_result_at else 0.0
                ),
            )
            self._last_result_at = completed_at

    def _worker(self):
        while self._running:
            with self._lock:
                enabled = bool(self.config.enabled)
                generation = self._generation
                source_type, source_value = self._source_type, self._source_value
                audio_source = self._effective_audio_source()
            if not enabled:
                self._wake.wait(.5)
                self._wake.clear()
                continue
            try:
                with self._lock:
                    self._state.update(status="loading", error="")
                self._load_model()
                source = self._open_source(audio_source, source_type, source_value)
                with self._lock:
                    if generation != self._generation or not self.config.enabled:
                        source.close()
                        continue
                    self._active_source = source
                    self._state.update(
                        status="listening",
                        audio_source=audio_source,
                        audio_device=self._selected_audio_device(),
                    )
                capture_frames = max(
                    self.SAMPLE_RATE // 10,
                    int(self.SAMPLE_RATE * float(self.config.capture_seconds)),
                )
                window_frames = max(
                    capture_frames,
                    int(self.SAMPLE_RATE * float(self.config.window_seconds)),
                )
                update_frames = max(
                    capture_frames,
                    int(self.SAMPLE_RATE * float(self.config.update_seconds)),
                )
                minimum_frames = max(
                    update_frames,
                    int(self.SAMPLE_RATE * float(self.config.minimum_seconds)),
                )
                rolling = np.empty(0, dtype=np.float32)
                captured_since_update = 0
                while self._running:
                    with self._lock:
                        if generation != self._generation or not self.config.enabled:
                            break
                    audio = source.read(capture_frames)
                    with self._lock:
                        if generation != self._generation or not self.config.enabled:
                            break
                    if audio.size < capture_frames // 2:
                        raise RuntimeError("audio stream ended unexpectedly")
                    rolling = np.concatenate((rolling, audio))[-window_frames:]
                    captured_since_update += audio.size
                    if rolling.size < minimum_frames or captured_since_update < update_frames:
                        continue
                    captured_since_update = 0
                    with self._lock:
                        self._state["status"] = "transcribing"
                    # Music often fails a speech-only VAD even when lyrics are
                    # clear, so system playback uses the complete rolling window.
                    self._recognize(
                        rolling.copy(),
                        vad_filter=audio_source != "system",
                        recent_seconds=float(self.config.display_seconds),
                    )
            except Exception as exc:
                with self._lock:
                    if self.config.enabled:
                        self._state.update(status="error", error=str(exc))
                self._wake.wait(1.0)
                self._wake.clear()
            finally:
                self._close_active_source()

    def close(self):
        self._running = False
        self._wake.set()
        self._thread.join(timeout=float(self.config.chunk_seconds) + 2.0)
        if self._thread.is_alive():
            self._close_active_source()
            self._thread.join(timeout=1.0)
