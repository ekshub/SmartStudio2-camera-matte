"""SmartStudio browser front-end/backend service.

This is the web equivalent of the Qt shell: a capture worker feeds the
ProcessingPipeline, the latest encoded frame is exposed as MJPEG, and the
browser controls are backed by JSON APIs.  It intentionally keeps only the
latest frame so a slow GPU cannot create an ever-growing latency queue.
"""
from __future__ import annotations

import argparse
import ast
import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file, send_from_directory

from src.audio.live_subtitles import LiveSubtitleEngine, list_audio_devices
from src.runtime.pipeline import ProcessingPipeline
from src.matting.rvm_engine import RVMEngine
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "web" / "templates"
DEFAULT_VIDEO = ROOT / "assets" / "demo" / "live_studio_test.mp4"
DEFAULT_BG_DIR = ROOT / "web" / "backgrounds"


class StudioRuntime:
    def __init__(self, source: str | None = None, camera: int | None = None, processing_scale: float = .4, person_prior: bool = False, inference_stride: int = 3):
        self.cfg = load_config(ROOT / "configs" / "default.yaml")
        self.cfg.rvm.checkpoint = str((ROOT / self.cfg.rvm.checkpoint).resolve())
        # The web shell is a live preview, so use the low-latency profile by
        # default. Full-resolution quality remains available with
        # --processing-scale 1.0 or the browser slider.
        self.cfg.runtime.person_prior = bool(person_prior)
        self.cfg.runtime.processing_scale = float(np.clip(processing_scale, .25, 1.0))
        # Camera mattes are displayed live; stale optical-flow keyframes make
        # hair and shoulders visibly shimmer. Always refresh the recurrent
        # matte on camera frames. Video playback may still use a stride.
        self.cfg.runtime.inference_stride = 1 if camera is not None else max(1, int(inference_stride))
        if camera is not None:
            # Use a larger inference canvas for camera quality. The previous
            # .4 default permanently discarded hair/shoulder detail before
            # the matte was generated.
            self.cfg.runtime.processing_scale = max(self.cfg.runtime.processing_scale, .75)
            self.cfg.rvm.variant = "resnet50"
            self.cfg.rvm.checkpoint = str((ROOT / "models" / "rvm_resnet50.pth").resolve())
            self.cfg.runtime.temporal_strength = max(.45, float(self.cfg.runtime.temporal_strength))
        self.cfg.runtime.preserve_source_detail = True
        self.cfg.runtime.person_prior_every_n = max(1, int(self.cfg.runtime.person_prior_every_n))
        # Keep the default live profile within a 33 ms/frame budget. Effects
        # remain switchable from the browser, but expensive full-frame glow and
        # shadow are opt-in.
        self.cfg.effects.enabled = False
        # Live profile keeps the source-resolution carrier untouched. The
        # offline/quality profile can re-enable harmonization through the API.
        self.cfg.harmonization.enabled = False
        self.cfg.runtime.inference_scale = 1.0
        self.pipeline = ProcessingPipeline(self.cfg)
        self.pipeline.set_inference_stride(self.cfg.runtime.inference_stride)
        if self.pipeline.person_prior is not None:
            # YOLO is a rescue prior, not the primary matte. Updating it every
            # fourth frame and using a smaller inference canvas keeps the live
            # path responsive while RVM recurrent state runs every frame.
            self.pipeline.person_prior.every_n = max(4, self.pipeline.person_prior.every_n)
            self.pipeline.person_prior.imgsz = 480
        self.lock = threading.RLock()
        self.frame_cond = threading.Condition(self.lock)
        self.pipeline_lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_frame: np.ndarray | None = None
        self.frame_id = 0
        self.processed = 0
        self.source_fps = 0.0
        self.processing_fps = 0.0
        self.last_latency_ms = 0.0
        self.last_mode = "starting"
        self.last_error = ""
        self.quality_guard = False
        self._slow_frames = 0
        self.running = True
        self._source_generation = 0
        self.source_type = "camera" if camera is not None else "video"
        self.source_path = str(camera if camera is not None else (source or DEFAULT_VIDEO))
        self.subtitles = LiveSubtitleEngine(self.cfg.subtitles, ROOT)
        self.subtitles.set_source(self.source_type, self.source_path)
        DEFAULT_BG_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_default_background()
        self.thread = threading.Thread(target=self._worker, name="smartstudio-web-worker", daemon=True)
        self.thread.start()

    @staticmethod
    def _ensure_default_background():
        p = DEFAULT_BG_DIR / "studio_gradient.jpg"
        if p.exists():
            return
        h, w = 720, 1280
        yy, xx = np.indices((h, w))
        bg = np.zeros((h, w, 3), np.uint8)
        bg[..., 0] = np.clip(55 + 45 * xx / w, 0, 255)
        bg[..., 1] = np.clip(35 + 55 * yy / h, 0, 255)
        bg[..., 2] = np.clip(110 + 50 * (1 - xx / w), 0, 255)
        cv2.circle(bg, (int(w * .78), int(h * .2)), int(min(w, h) * .16), (170, 100, 45), -1)
        cv2.imwrite(str(p), bg)

    def _open_capture(self):
        with self.lock:
            typ, path = self.source_type, self.source_path
        cap = cv2.VideoCapture(int(path) if typ == "camera" else str(path))
        if typ == "camera":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def _worker(self):
        cap = None
        generation = -1
        source_frames = 0
        source_t0 = time.perf_counter()
        proc_t0 = time.perf_counter()
        media_fps = 0.0
        media_frame = 0
        media_t0 = time.perf_counter()
        while self.running:
            with self.lock:
                current_generation = self._source_generation
            if cap is None or generation != current_generation:
                if cap is not None:
                    cap.release()
                cap = self._open_capture()
                generation = current_generation
                source_frames = 0
                self.processed = 0
                source_t0 = proc_t0 = time.perf_counter()
                media_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap is not None else 0.0
                media_fps = media_fps if 1.0 <= media_fps <= 240.0 else 30.0
                media_frame = 0
                media_t0 = time.perf_counter()
                with self.pipeline_lock:
                    self.pipeline.reset_state()
            if cap is None or not cap.isOpened():
                time.sleep(.2)
                continue
            ok, frame = cap.read()
            if not ok:
                with self.lock:
                    typ = self.source_type
                if typ == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    media_frame = 0
                    media_t0 = time.perf_counter()
                    time.sleep(.03)
                    continue
                time.sleep(.1)
                continue
            with self.lock:
                typ = self.source_type
            if typ == "video":
                deadline = media_t0 + media_frame / media_fps
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                media_frame += 1
            source_frames += 1
            now = time.perf_counter()
            if now - source_t0 >= 1:
                self.source_fps = source_frames / (now - source_t0)
                source_frames, source_t0 = 0, now
            start = time.perf_counter()
            try:
                with self.pipeline_lock:
                    image, alpha, mode, latency = self.pipeline.process(frame)
                ok_jpg, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok_jpg:
                    continue
                elapsed = time.perf_counter() - start
                with self.frame_cond:
                    self.latest_frame = image
                    self.latest_jpeg = buf.tobytes()
                    self.frame_id += 1
                    self.processed += 1
                    self.last_latency_ms = float(latency)
                    self.last_mode = mode
                    self.last_error = ""
                    if latency > 33.3:
                        self._slow_frames += 1
                    else:
                        self._slow_frames = max(0, self._slow_frames - 1)
                    if self._slow_frames >= 5:
                        changed = False
                        if typ != "camera" and self.pipeline.propagator.stride < 3:
                            self.pipeline.set_inference_stride(3)
                            changed = True
                        # Never trade away the camera quality canvas to chase a
                        # 30 FPS target: dropping from .75 back to .4 is the
                        # main reason hair edges become visibly unstable.
                        min_scale = .75 if typ == "camera" else .40
                        if self.pipeline.processing_scale > min_scale:
                            self.pipeline.set_processing_scale(max(min_scale, self.pipeline.processing_scale - .05))
                            self.cfg.runtime.processing_scale = self.pipeline.processing_scale
                            changed = True
                        if self.cfg.runtime.person_prior:
                            self.cfg.runtime.person_prior = False
                            self.pipeline.person_prior = None
                            changed = True
                        if self.cfg.effects.enabled and (self.cfg.effects.glow or self.cfg.effects.shadow):
                            self.cfg.effects.glow = self.cfg.effects.shadow = False
                            changed = True
                        self.quality_guard = self.quality_guard or changed
                        self._slow_frames = 0
                    if time.perf_counter() - proc_t0 >= 1:
                        self.processing_fps = self.processed / (time.perf_counter() - proc_t0)
                        self.processed, proc_t0 = 0, time.perf_counter()
                    self.frame_cond.notify_all()
            except Exception as exc:
                with self.lock:
                    self.last_mode = "error"
                    self.last_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                time.sleep(.05)
        if cap is not None:
            cap.release()

    def set_source(self, source_type: str, value):
        if source_type not in {"video", "camera"}:
            raise ValueError("source_type must be video or camera")
        if source_type == "camera":
            value = int(value)
        else:
            p = Path(str(value)).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(str(p))
            value = str(p)
        # Source changes can happen from the browser after startup. Select
        # the quality model and canvas here as well as in __init__, otherwise
        # switching the UI from video to camera silently kept MobileNetV3.
        if source_type == "camera":
            self.cfg.runtime.processing_scale = max(float(self.cfg.runtime.processing_scale), .75)
            self.cfg.runtime.inference_stride = 1
            self.cfg.runtime.temporal_strength = max(.45, float(self.cfg.runtime.temporal_strength))
            self.cfg.rvm.variant = "resnet50"
            self.cfg.rvm.checkpoint = str((ROOT / "models" / "rvm_resnet50.pth").resolve())
        else:
            self.cfg.rvm.variant = "mobilenetv3"
            self.cfg.rvm.checkpoint = str((ROOT / "models" / "rvm_mobilenetv3.pth").resolve())
        with self.pipeline_lock:
            # Recreate only the matting engine; the rest of the pipeline and
            # UI settings remain intact. State is reset so recurrent tensors
            # from the previous source/model cannot contaminate the first frame.
            self.pipeline.rvm = RVMEngine(
                self.cfg.rvm.checkpoint, fp16=self.cfg.rvm.fp16,
                downsample_ratio=self.cfg.rvm.downsample_ratio,
                inference_scale=self.cfg.runtime.inference_scale,
                variant=self.cfg.rvm.variant)
            self.pipeline.set_processing_scale(self.cfg.runtime.processing_scale)
            self.pipeline.set_inference_stride(self.cfg.runtime.inference_stride)
            self.pipeline.reset_state()
        with self.lock:
            self.source_type, self.source_path = source_type, str(value)
            self._source_generation += 1
        self.subtitles.set_source(source_type, value)

    def list_backgrounds(self):
        # Browser/API paths must use POSIX separators even on Windows; the
        # server resolves them back to local paths when applying a background.
        return sorted(p.relative_to(ROOT).as_posix() for p in DEFAULT_BG_DIR.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    @staticmethod
    def list_audio_devices():
        return list_audio_devices()

    def apply_config(self, data: dict):
        changed_bg = False
        with self.pipeline_lock:
            if "processing_scale" in data:
                value = float(np.clip(float(data["processing_scale"]), .25, 1.0))
                self.pipeline.set_processing_scale(value)
                self.cfg.runtime.processing_scale = value
            if "inference_stride" in data:
                self.pipeline.set_inference_stride(max(1, int(data["inference_stride"])))
            if "preserve_source_detail" in data:
                self.cfg.runtime.preserve_source_detail = bool(data["preserve_source_detail"])
            if "green_screen_optimized" in data:
                self.cfg.matting.green_screen_optimized = bool(data["green_screen_optimized"])
            if "harmonization_enabled" in data:
                self.cfg.harmonization.enabled = bool(data["harmonization_enabled"])
            if "person_prior" in data:
                self.cfg.runtime.person_prior = bool(data["person_prior"])
                # The model object is intentionally rebuilt only when toggled.
                self.pipeline.person_prior = None
                if self.cfg.runtime.person_prior:
                    from src.matting.person_prior import PersonSegmentationPrior
                    self.pipeline.person_prior = PersonSegmentationPrior(self.cfg.runtime.person_prior_weights, self.cfg.runtime.person_prior_conf, imgsz=480, every_n=max(4, self.cfg.runtime.person_prior_every_n))
            effects = data.get("effects", {})
            for key in ("enabled", "outline", "glow", "shadow", "face_effect_enabled"):
                if key in effects:
                    setattr(self.cfg.effects, key, bool(effects[key]))
            for key in ("outline_width", "glow_radius", "shadow_blur", "shadow_dx", "shadow_dy", "face_effect_interval", "face_effect_max_faces"):
                if key in effects:
                    setattr(self.cfg.effects, key, int(effects[key]))
            for key in ("outline_opacity", "glow_opacity", "shadow_opacity", "text_opacity", "face_effect_opacity"):
                if key in effects:
                    setattr(self.cfg.effects, key, float(np.clip(float(effects[key]), 0, 1)))
            if "text" in effects:
                self.cfg.effects.text = str(effects["text"])[:120]
            if "face_accessory" in effects:
                accessory = str(effects["face_accessory"]).lower()
                if accessory not in {"none", "glasses", "sunglasses", "hat"}:
                    raise ValueError("unsupported face accessory")
                self.cfg.effects.face_accessory = accessory
            self.pipeline.face_effects.interval = max(1, int(self.cfg.effects.face_effect_interval))
            self.pipeline.face_effects.max_faces = max(1, int(self.cfg.effects.face_effect_max_faces))
            if "background" in data:
                bg = data["background"]
                kind = str(bg.get("kind", self.cfg.background.kind))
                # Solid backgrounds do not have a file value.  In particular,
                # the browser omits `value` when the disabled image selector
                # is serialized; do not inherit an old image path here.
                value = (
                    bg.get("value")
                    if kind == "solid"
                    else bg.get("value", self.cfg.background.value)
                )
                if kind not in {"solid", "blur", "transparent", "image", "video"}:
                    raise ValueError("unsupported background kind")
                if kind in {"image", "video"}:
                    # A selected image is represented by a relative option in
                    # the browser, while the API stores an absolute path. If
                    # the browser sends an empty value (for example after a
                    # page refresh), retain the active file instead of
                    # resolving an empty path to the project directory.
                    if value is None or (isinstance(value, str) and not value.strip()):
                        current = Path(str(self.cfg.background.value)).expanduser()
                        if self.cfg.background.kind == kind and current.is_file():
                            value = str(current)
                        else:
                            raise ValueError("请选择有效的背景文件")
                    p = Path(str(value)).expanduser()
                    if not p.is_absolute():
                        p = (ROOT / p).resolve()
                    if not p.is_file():
                        raise FileNotFoundError(str(p))
                    value = str(p)
                elif kind == "solid":
                    # The browser may submit the JSON-rendered color as a
                    # string (e.g. "[30, 60, 100]"). Normalize it back to a
                    # validated BGR tuple before numpy receives it.
                    # A solid background has no file selection.  Browsers
                    # represent the image-only <select> value as an empty
                    # string when the current config contains a color array;
                    # preserve the existing color in that case.
                    if value is None or (isinstance(value, str) and not value.strip()):
                        value = (
                            self.cfg.background.value
                            if self.cfg.background.kind == "solid"
                            else (40, 40, 40)
                        )
                    if isinstance(value, str):
                        try:
                            value = ast.literal_eval(value)
                        except (SyntaxError, ValueError):
                            raise ValueError("solid background must be a 3-channel color")
                    if not isinstance(value, (list, tuple)) or len(value) != 3:
                        raise ValueError("solid background must be a 3-channel color")
                    value = tuple(int(np.clip(float(channel), 0, 255)) for channel in value)
                old = (self.cfg.background.kind, str(self.cfg.background.value))
                self.cfg.background.kind, self.cfg.background.value = kind, value
                changed_bg = old != (kind, str(value))
                if changed_bg:
                    self.pipeline.bg.close()
                    from src.compositing.background import BackgroundProvider
                    self.pipeline.bg = BackgroundProvider(kind, value, self.cfg.background.loop)
        subtitle_data = data.get("subtitles")
        if subtitle_data is not None:
            old_enabled = bool(self.cfg.subtitles.enabled)
            old_language = self.cfg.subtitles.language
            old_audio_source = self.cfg.subtitles.audio_source
            old_microphone = self.cfg.subtitles.microphone_device
            old_system_audio = self.cfg.subtitles.system_audio_device
            enabled = bool(subtitle_data.get("enabled", old_enabled))
            language = str(subtitle_data.get("language", old_language))
            audio_source = str(subtitle_data.get("audio_source", old_audio_source))
            microphone = str(subtitle_data.get("microphone_device", old_microphone))
            system_audio = str(subtitle_data.get("system_audio_device", old_system_audio))
            if audio_source == "video" and self.source_type != "video":
                raise ValueError("视频音轨只能用于视频输入，请先选择视频或改用麦克风/电脑音频")
            changed = any((
                enabled != old_enabled,
                language != old_language,
                audio_source != old_audio_source,
                microphone != old_microphone,
                system_audio != old_system_audio,
            ))
            if changed:
                restart_audio = enabled and any((
                    not old_enabled,
                    language != old_language,
                    audio_source != old_audio_source,
                    microphone != old_microphone,
                    system_audio != old_system_audio,
                ))
                uses_video_audio = audio_source == "video" or (
                    audio_source == "auto" and self.source_type == "video"
                )
                if restart_audio and uses_video_audio:
                    # Start video and its FFmpeg audio decoder from zero together.
                    with self.lock:
                        self._source_generation += 1
                self.subtitles.configure(
                    enabled=enabled,
                    language=language,
                    audio_source=audio_source,
                    microphone_device=microphone,
                    system_audio_device=system_audio,
                )
        return self.config_dict()

    def config_dict(self):
        return {
            "source": {"type": self.source_type, "value": self.source_path},
            "processing_scale": float(self.cfg.runtime.processing_scale),
            "inference_stride": int(self.cfg.runtime.inference_stride),
            "preserve_source_detail": bool(getattr(self.cfg.runtime, "preserve_source_detail", True)),
            "harmonization_enabled": bool(self.cfg.harmonization.enabled),
            "green_screen_optimized": bool(getattr(self.cfg.matting, "green_screen_optimized", True)),
            "person_prior": bool(self.cfg.runtime.person_prior),
                "background": {
                    "kind": self.cfg.background.kind,
                    "value": (
                        list(self.cfg.background.value)
                        if self.cfg.background.kind == "solid"
                        and isinstance(self.cfg.background.value, (list, tuple))
                        else str(self.cfg.background.value)
                    ),
                },
            "effects": {k: getattr(self.cfg.effects, k) for k in ("enabled", "outline", "glow", "shadow", "outline_width", "glow_radius", "shadow_blur", "shadow_dx", "shadow_dy", "outline_opacity", "glow_opacity", "shadow_opacity", "text", "text_opacity", "face_effect_enabled", "face_accessory", "face_effect_opacity", "face_effect_interval", "face_effect_max_faces")},
            "subtitles": {
                "enabled": bool(self.cfg.subtitles.enabled),
                "language": self.cfg.subtitles.language,
                "model": self.cfg.subtitles.model,
                "audio_source": self.cfg.subtitles.audio_source,
                "microphone_device": self.cfg.subtitles.microphone_device,
                "system_audio_device": self.cfg.subtitles.system_audio_device,
            },
        }

    def status(self):
        with self.lock:
            result = {"running": self.running, "frame_id": self.frame_id, "source_fps": round(self.source_fps, 2), "processing_fps": round(self.processing_fps, 2), "latency_ms": round(self.last_latency_ms, 2), "mode": self.last_mode, "error": self.last_error, "processing_scale": float(self.cfg.runtime.processing_scale), "inference_stride": int(self.cfg.runtime.inference_stride), "person_prior": bool(self.cfg.runtime.person_prior), "effects_enabled": bool(self.cfg.effects.enabled), "quality_guard": bool(self.quality_guard), "source": {"type": self.source_type, "value": self.source_path}}
        result["subtitle"] = self.subtitles.status()
        return result

    def wait_frame(self, last_id: int, timeout: float = 2.0):
        with self.frame_cond:
            self.frame_cond.wait_for(lambda: self.frame_id != last_id or not self.running, timeout=timeout)
            return self.frame_id, self.latest_jpeg

    def close(self):
        self.running = False
        self.subtitles.close()
        with self.frame_cond:
            self.frame_cond.notify_all()
        self.pipeline.bg.close()


def create_app(runtime: StudioRuntime | None = None):
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
    rt = runtime or StudioRuntime()
    app.config["studio_runtime"] = rt

    @app.get("/")
    def index():
        return render_template("index.html", config=rt.config_dict(), backgrounds=rt.list_backgrounds())

    @app.get("/backgrounds/<path:name>")
    def background_asset(name):
        """Serve the curated local scene gallery to the browser."""
        return send_from_directory(DEFAULT_BG_DIR, name)

    @app.get("/video_feed")
    def video_feed():
        def stream():
            last = -1
            while rt.running:
                frame_id, jpg = rt.wait_frame(last)
                if jpg is None:
                    continue
                last = frame_id
                yield b"--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Id: " + str(frame_id).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/get_frame")
    def get_frame():
        _, jpg = rt.wait_frame(-1, timeout=.2)
        return Response(jpg or b"", mimetype="image/jpeg")

    @app.get("/api/status")
    def status():
        return jsonify(rt.status())

    @app.get("/api/config")
    def get_config():
        return jsonify(rt.config_dict())

    @app.get("/api/audio/devices")
    def audio_devices():
        return jsonify(rt.list_audio_devices())

    @app.post("/api/config")
    def update_config():
        try:
            return jsonify({"ok": True, "config": rt.apply_config(request.get_json(force=True) or {})})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/source")
    def update_source():
        try:
            data = request.get_json(force=True) or {}
            rt.set_source(str(data.get("type", "video")), data.get("value"))
            return jsonify({"ok": True, "source": {"type": rt.source_type, "value": rt.source_path}})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/reset")
    def reset():
        with rt.pipeline_lock:
            rt.pipeline.reset_state()
        return jsonify({"ok": True})

    @app.post("/api/snapshot")
    def snapshot():
        with rt.lock:
            frame = None if rt.latest_frame is None else rt.latest_frame.copy()
        if frame is None:
            return jsonify({"ok": False, "error": "no frame yet"}), 409
        out_dir = ROOT / "output" / "web_snapshots"; out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"smartstudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), frame)
        return send_file(path, mimetype="image/jpeg", as_attachment=True, download_name=path.name)

    return app


def main():
    parser = argparse.ArgumentParser(description="SmartStudio browser studio")
    parser.add_argument("--source", default=str(DEFAULT_VIDEO), help="video path")
    parser.add_argument("--camera", type=int, default=None, help="camera index; overrides --source")
    parser.add_argument("--processing-scale", type=float, default=.4, help="live processing scale (0.25-1.0; default 0.4 with key-frame propagation)")
    parser.add_argument("--inference-stride", type=int, default=3, help="RVM key-frame stride; 3 propagates intermediate frames")
    parser.add_argument("--person-prior", action="store_true", help="enable YOLO rescue prior (slower)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    runtime = StudioRuntime(source=args.source, camera=args.camera, processing_scale=args.processing_scale, person_prior=args.person_prior, inference_stride=args.inference_stride)
    app = create_app(runtime)
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
