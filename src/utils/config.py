from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class RVMConfig:
    variant: str = "mobilenetv3"
    checkpoint: str = "models/rvm_mobilenetv3.pth"
    fp16: bool = True
    downsample_ratio: float = 0.25

@dataclass
class RuntimeConfig:
    target_fps: float = 30.0
    adaptive_quality: bool = True
    inference_scale: float = 1.0
    inference_stride: int = 1
    preserve_source_detail: bool = False
    # Scale used for the complete matting/compositing pipeline.  The final
    # frame is always restored to the input dimensions.  1.0 preserves full
    # quality; 0.5 is the low-latency profile for 720p cameras.
    processing_scale: float = 1.0
    temporal_stabilization: bool = True
    temporal_strength: float = 0.28
    connectivity_repair: bool = True
    connectivity_strength: float = 0.35
    person_prior: bool = False
    person_prior_weights: str = "yolo11n-seg.pt"
    person_prior_conf: float = 0.15
    person_prior_every_n: int = 2

@dataclass
class MattingConfig:
    mode: str = "auto"
    screen_low_threshold: float = 0.45
    screen_high_threshold: float = 0.65
    hybrid_max_weight: float = 0.8
    green_screen_optimized: bool = True

@dataclass
class SpillConfig:
    enabled: bool = True
    strength: float = 0.6

@dataclass
class HarmonizationConfig:
    enabled: bool = True
    strength: float = 0.15
    temporal_ema: float = 0.1

@dataclass
class BackgroundConfig:
    kind: str = "solid"  # solid, image, video, blur, transparent
    value: Any = (30, 60, 100)
    loop: bool = True

@dataclass
class EffectsConfig:
    enabled: bool = True
    outline: bool = False
    outline_width: int = 3
    outline_opacity: float = 0.8
    glow: bool = False
    glow_radius: float = 15.0
    glow_opacity: float = 0.4
    shadow: bool = False
    shadow_dx: int = 8
    shadow_dy: int = 8
    shadow_blur: float = 12.0
    shadow_opacity: float = 0.4
    text: str = ""
    text_x: int = 20
    text_y: int = 40
    text_size: float = 1.0
    logo: str = ""
    logo_x: int = 20
    logo_y: int = 20
    logo_scale: float = 1.0
    logo_opacity: float = 1.0
    outline_color: tuple = (255, 255, 255)
    glow_color: tuple = (255, 255, 255)
    shadow_color: tuple = (0, 0, 0)
    text_opacity: float = 1.0
    foreground_scale: float = 1.0
    foreground_x: int = 0
    foreground_y: int = 0
    # Tracked face accessories are independent from the expensive glow/shadow
    # effects so they can remain enabled in the live 30 FPS profile.
    face_effect_enabled: bool = False
    face_accessory: str = "none"
    face_effect_opacity: float = 0.95
    face_effect_interval: int = 2
    face_effect_max_faces: int = 2

@dataclass
class SubtitleConfig:
    enabled: bool = False
    model: str = "models/faster-whisper-large-v3-turbo"
    device: str = "auto"
    compute_type: str = "int8_float16"
    language: str = "auto"
    audio_source: str = "auto"
    microphone_device: str = "default"
    system_audio_device: str = "default"
    capture_seconds: float = 0.5
    window_seconds: float = 10.0
    minimum_seconds: float = 1.0
    display_seconds: float = 3.5
    update_seconds: float = 1.0
    chunk_seconds: float = 2.5  # Legacy shutdown timeout fallback.
    hold_seconds: float = 4.0

@dataclass
class AppConfig:
    rvm: RVMConfig = field(default_factory=RVMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    matting: MattingConfig = field(default_factory=MattingConfig)
    spill: SpillConfig = field(default_factory=SpillConfig)
    harmonization: HarmonizationConfig = field(default_factory=HarmonizationConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    subtitles: SubtitleConfig = field(default_factory=SubtitleConfig)

def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parents[2] / p
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return AppConfig(RVMConfig(**data.get("rvm", {})), RuntimeConfig(**data.get("runtime", {})),
                     MattingConfig(**data.get("matting", {})), SpillConfig(**data.get("spill", {})),
                     HarmonizationConfig(**data.get("harmonization", {})),
                     BackgroundConfig(**data.get("background", {})), EffectsConfig(**data.get("effects", {})),
                     SubtitleConfig(**data.get("subtitles", {})))
