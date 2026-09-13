"""Lightweight, tracked face accessories for the live preview.

The detector runs on a small RGB image only every ``interval`` frames.  The
last landmarks are exponentially smoothed and reused between detections, so
the accessory layer does not add a full face-mesh inference to every frame.
"""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class _FaceTrack:
    points: np.ndarray
    age: int = 0


class FaceEffectEngine:
    """MediaPipe Face Mesh backed accessory renderer.

    The class is deliberately optional: if MediaPipe is not installed, face
    effects stay disabled and the regular matting pipeline continues to work.
    """

    def __init__(self, max_faces: int = 2, interval: int = 2, smooth: float = 0.35):
        self.max_faces = max(1, int(max_faces))
        self.interval = max(1, int(interval))
        self.smooth = float(np.clip(smooth, 0.0, 0.95))
        self._counter = 0
        self._tracks: list[_FaceTrack] = []
        self._track_gray = None
        self._mesh = None
        self.available = False
        self._assets = {}
        asset_dir = Path(__file__).resolve().parents[2] / "assets" / "face_effects"
        for name in ("glasses", "sunglasses", "hat"):
            self._assets[name] = self._load_asset(asset_dir / f"openmoji_{name}.png")
        try:
            import mediapipe as mp

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_faces,
                refine_landmarks=False,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.available = True
        except Exception:
            self._mesh = None

    @staticmethod
    def _load_asset(path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
        # Crop transparent margins once. This keeps the target width tied to
        # the visible artwork instead of the 618x618 source canvas.
        alpha = image[..., 3]
        ys, xs = np.where(alpha > 2)
        if len(xs):
            image = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return image

    def close(self):
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None

    def reset(self):
        self._counter = 0
        self._tracks.clear()
        self._track_gray = None

    def _predict_tracks(self, frame: np.ndarray):
        """Advance landmarks every frame with sparse optical flow.

        MediaPipe is used as a correction keyframe; LK flow fills the frames
        between corrections so stickers do not visibly lag behind fast motion.
        """
        if not self._tracks:
            self._track_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._track_gray is None or self._track_gray.shape != gray.shape:
            self._track_gray = gray
            return
        for track in self._tracks:
            p0 = track.points.reshape(-1, 1, 2).astype(np.float32)
            p1, status, _ = cv2.calcOpticalFlowPyrLK(
                self._track_gray, gray, p0, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
            )
            if p1 is None or status is None:
                track.age += 1
                continue
            good = status.reshape(-1).astype(bool)
            if int(good.sum()) < max(40, int(len(good) * .55)):
                track.age += 1
                continue
            updated = track.points.copy()
            updated[good] = p1.reshape(-1, 2)[good]
            track.points = updated.astype(np.float32)
            track.age += 1
        self._track_gray = gray

    @staticmethod
    def _points(face, width: int, height: int) -> np.ndarray:
        return np.asarray([(p.x * width, p.y * height) for p in face.landmark], dtype=np.float32)

    def _detect(self, frame: np.ndarray) -> list[np.ndarray]:
        if not self.available or self._mesh is None:
            return []
        h, w = frame.shape[:2]
        # Face mesh is much faster and more stable at a bounded input size.
        scale = min(1.0, 640.0 / max(w, h))
        if scale < 1.0:
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []
        sw, sh = small.shape[1], small.shape[0]
        factor_x, factor_y = w / sw, h / sh
        faces = []
        for face in result.multi_face_landmarks[: self.max_faces]:
            p = self._points(face, sw, sh)
            p[:, 0] *= factor_x
            p[:, 1] *= factor_y
            faces.append(p)
        return faces

    def _update_tracks(self, detections: list[np.ndarray]):
        if not detections:
            for track in self._tracks:
                track.age += 1
            self._tracks = [t for t in self._tracks if t.age <= self.interval * 3]
            return
        if not self._tracks:
            self._tracks = [_FaceTrack(p.copy()) for p in detections]
            return
        # Match faces by nose/face-center distance.  This is adequate for the
        # short interval between mesh evaluations and avoids a heavy tracker.
        old_centers = [t.points[1] if len(t.points) > 1 else t.points.mean(0) for t in self._tracks]
        used = set()
        new_tracks: list[_FaceTrack] = []
        for points in detections:
            center = points[1] if len(points) > 1 else points.mean(0)
            candidates = [(float(np.linalg.norm(center - old_centers[i])), i) for i in range(len(self._tracks)) if i not in used]
            if candidates:
                distance, idx = min(candidates)
            else:
                distance, idx = 1e9, -1
            if idx >= 0 and distance < max(80.0, 0.6 * self._face_width(points)):
                previous = self._tracks[idx].points
                # Keep correction smoothing modest: optical flow already
                # provides temporal continuity, while heavy EMA causes lag.
                points = self.smooth * previous + (1.0 - self.smooth) * points
                used.add(idx)
            new_tracks.append(_FaceTrack(points.astype(np.float32), 0))
        self._tracks = new_tracks[: self.max_faces]
        if self._track_gray is None:
            self._track_gray = None

    @staticmethod
    def _face_width(points: np.ndarray) -> float:
        if len(points) <= 263:
            return 120.0
        return float(np.linalg.norm(points[263] - points[33]))

    @staticmethod
    def _blend(dst: np.ndarray, layer: np.ndarray, alpha: np.ndarray):
        a = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
        dst[:] = np.clip(dst.astype(np.float32) * (1.0 - a) + layer.astype(np.float32) * a, 0, 255).astype(np.uint8)

    def _overlay_asset(self, frame: np.ndarray, asset: np.ndarray, center, width: float,
                       angle: float, opacity: float, visible: np.ndarray | None = None):
        """Scale/rotate a BGRA sticker and alpha-composite it onto frame."""
        if asset is None or width <= 1:
            return
        ah, aw = asset.shape[:2]
        scale = float(width) / max(1, aw)
        nw, nh = max(1, int(round(aw * scale))), max(1, int(round(ah * scale)))
        sticker = cv2.resize(asset, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        mat = cv2.getRotationMatrix2D((nw * .5, nh * .5), angle, 1.0)
        cos, sin = abs(mat[0, 0]), abs(mat[0, 1])
        rw, rh = max(1, int(round(nh * sin + nw * cos))), max(1, int(round(nh * cos + nw * sin)))
        mat[0, 2] += rw * .5 - nw * .5
        mat[1, 2] += rh * .5 - nh * .5
        sticker = cv2.warpAffine(sticker, mat, (rw, rh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        x0, y0 = int(round(float(center[0]) - rw * .5)), int(round(float(center[1]) - rh * .5))
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(frame.shape[1], x0 + rw), min(frame.shape[0], y0 + rh)
        if x1 >= x2 or y1 >= y2:
            return
        sx1, sy1, sx2, sy2 = x1 - x0, y1 - y0, x2 - x0, y2 - y0
        patch = sticker[sy1:sy2, sx1:sx2]
        alpha = patch[..., 3].astype(np.float32) * float(np.clip(opacity, 0, 1))
        if visible is not None:
            vm = visible[y1:y2, x1:x2]
            if vm.shape == alpha.shape:
                alpha *= np.clip(vm.astype(np.float32), 0, 1)
        roi = frame[y1:y2, x1:x2]
        a = (alpha / 255.0)[..., None]
        roi[:] = np.clip(roi.astype(np.float32) * (1 - a) + patch[..., :3].astype(np.float32) * a, 0, 255).astype(np.uint8)

    def _draw_glasses(self, frame: np.ndarray, points: np.ndarray, style: str, opacity: float, visible: np.ndarray | None):
        if len(points) <= 386:
            return
        # MediaPipe indices: 33/133 left eye corners, 362/263 right eye
        # corners.  These anchors remain stable under moderate head rotation.
        left = (points[33] + points[133]) * 0.5
        right = (points[362] + points[263]) * 0.5
        center = (left + right) * 0.5
        eye_distance = float(np.linalg.norm(right - left))
        if not np.isfinite(eye_distance) or eye_distance < 12:
            return
        angle = math.degrees(math.atan2(float(right[1] - left[1]), float(right[0] - left[0])))
        asset = self._assets.get(style)
        if asset is not None:
            # The sticker is anchored by face landmarks. Do not clip it to the
            # person alpha: accessories such as hat brims intentionally extend
            # beyond the silhouette (and are already restricted to a face).
            self._overlay_asset(frame, asset, center, eye_distance * 2.15,
                                -angle, opacity, None)
            return
        width = eye_distance * 0.58
        height = eye_distance * 0.34
        thickness = max(2, int(round(eye_distance * 0.035)))
        color = (28, 28, 28) if style == "sunglasses" else (45, 105, 170)
        lens_alpha = 190 if style == "sunglasses" else 35
        layer = np.zeros_like(frame)
        mask = np.zeros(frame.shape[:2], np.uint8)

        def rotated_ellipse(c, axes, fill=False):
            cv2.ellipse(layer, tuple(np.round(c).astype(int)), tuple(np.round(axes).astype(int)), angle, 0, 360, color, -1 if fill else thickness, cv2.LINE_AA)
            if fill:
                cv2.ellipse(mask, tuple(np.round(c).astype(int)), tuple(np.round(axes).astype(int)), angle, 0, 360, lens_alpha, -1, cv2.LINE_AA)

        left_lens = center - (right - left) * 0.26
        right_lens = center + (right - left) * 0.26
        rotated_ellipse(left_lens, (width, height), style == "sunglasses")
        rotated_ellipse(right_lens, (width, height), style == "sunglasses")
        cv2.line(layer, tuple(np.round(left_lens + (right - left) * 0.18).astype(int)), tuple(np.round(right_lens - (right - left) * 0.18).astype(int)), color, thickness, cv2.LINE_AA)
        # Temple arms extend slightly beyond the outer corners.
        direction = (right - left) / max(eye_distance, 1.0)
        cv2.line(layer, tuple(np.round(left_lens - direction * width * 0.9).astype(int)), tuple(np.round(left_lens - direction * width * 0.2).astype(int)), color, thickness, cv2.LINE_AA)
        cv2.line(layer, tuple(np.round(right_lens + direction * width * 0.2).astype(int)), tuple(np.round(right_lens + direction * width * 0.9).astype(int)), color, thickness, cv2.LINE_AA)
        line_mask = np.any(layer != 0, axis=2)
        mask[line_mask] = np.maximum(mask[line_mask], int(255 * np.clip(opacity, 0, 1)))
        if style == "sunglasses":
            mask = np.maximum(mask, (cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * int(210 * np.clip(opacity, 0, 1)))
        if visible is not None and visible.shape == mask.shape:
            mask = (mask.astype(np.float32) * np.clip(visible.astype(np.float32), 0, 1)).astype(np.uint8)
        self._blend(frame, layer, mask)

    def _draw_hat(self, frame: np.ndarray, points: np.ndarray, opacity: float, visible: np.ndarray | None):
        if len(points) <= 454:
            return
        # Face top/width are stable under moderate pose changes. Move the
        # sticker slightly above the forehead so the brim does not cover eyes.
        face_left, face_right = points[234], points[454]
        top = np.mean(points[[10, 151, 9]], axis=0)
        face_width = float(np.linalg.norm(face_right - face_left))
        if not np.isfinite(face_width) or face_width < 20:
            return
        center = np.array([(face_left[0] + face_right[0]) * .5,
                           top[1] - face_width * .34], dtype=np.float32)
        left = (points[33] + points[133]) * .5
        right = (points[362] + points[263]) * .5
        angle = math.degrees(math.atan2(float(right[1] - left[1]), float(right[0] - left[0])))
        self._overlay_asset(frame, self._assets.get("hat"), center, face_width * 1.55,
                            -angle, opacity, None)

    def apply(self, frame: np.ndarray, alpha: np.ndarray | None, style: str = "glasses", opacity: float = 0.95) -> np.ndarray:
        """Return a frame with tracked glasses/sunglasses over detected faces."""
        if not self.available or style in {"", "none", None} or frame is None:
            return frame
        self._counter += 1
        self._predict_tracks(frame)
        if self._counter % self.interval == 1 or not self._tracks:
            self._update_tracks(self._detect(frame))
        for track in self._tracks:
            if str(style) == "hat":
                self._draw_hat(frame, track.points, opacity, alpha)
            else:
                self._draw_glasses(frame, track.points, str(style), opacity, alpha)
        return frame
