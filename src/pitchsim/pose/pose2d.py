"""
2D pose for broadcast pitching clips.

Tool choice: RTMPose via ``rtmlib`` (ONNX, CPU).  Why: it is multi-person (a broadcast frame has
batter, umpire, catcher; MediaPipe Pose is single-person and would lock onto whichever it likes),
so we detect everyone and pick the pitcher ourselves (largest person near frame centre).
Tried on Ohtani clips only with rtmlib (MediaPipe not benchmarked: single-person was the disqualifier).
Two lessons from the first runs: pose-on-everyone is ~1 s/frame with the crowd in view, so we
detect, pick the pitcher's box, then pose that one box; and tracking must be frame-to-frame (nearest
previous box), because a static 'most central' rule hops onto the catcher mid-delivery.

Keypoints are COCO-17: 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder,
7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee,
15 l_ankle, 16 r_ankle.

cv2 / rtmlib are imported lazily so screen.py, release.py and tests need only numpy.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Pose2D:
    xy: np.ndarray      # (F, 17, 2) pixels
    conf: np.ndarray    # (F, 17) 0..1
    fps: float
    size: tuple[int, int]  # (width_px, height_px)


def save_pose2d(p: Pose2D, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xy=p.xy, conf=p.conf, fps=p.fps, size=np.array(p.size))


def load_pose2d(path) -> Pose2D:
    z = np.load(path)
    return Pose2D(z["xy"], z["conf"], float(z["fps"]), (int(z["size"][0]), int(z["size"][1])))


def _pick_pitcher(boxes: np.ndarray, w: int, h: int) -> int:
    """Index of the person with the best (area, centredness) score.  Why: in centre-field view the
    pitcher is the big figure near the middle; the batter/umpire are partly hidden or off-axis."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    return int(np.argmax(area * np.exp(-(((cx - w / 2) / (0.25 * w)) ** 2))))


def _nearest(boxes: np.ndarray, prev: np.ndarray, w: int) -> int:
    """Index of the box whose centre is closest to the previously chosen box.  Why: the pitcher
    does not teleport between frames, and a static 'most central' rule can hop to the catcher or
    batter mid-delivery (seen: wrist x jumped 150 px in one frame)."""
    c = (boxes[:, :2] + boxes[:, 2:]) / 2
    pc = (prev[:2] + prev[2:]) / 2
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    d = np.linalg.norm(c - pc, axis=1) + 1e6 * (area < 0.3 * (prev[2] - prev[0]) * (prev[3] - prev[1]))
    return int(np.argmin(d))


_model = None
CROP = (0.1, 0.9)  # fraction of frame width kept for inference


def run_pose2d(video_path) -> Pose2D:
    """Run RTMPose on every frame of an mp4; frames with no person get conf 0."""
    import cv2
    from rtmlib import Body

    global _model
    if _model is None:
        # 'balanced' = rtmdet-m + rtmpose-m; ONNX on CPU (40 clips only, speed is fine).
        _model = Body(mode="balanced", to_openpose=False, backend="onnxruntime", device="cpu")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    xy, conf, prev = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Crop to the middle 80% of the width: the stands hold ~70 detectable people (0.8 s/frame);
        # the pitcher is central in centre-field view.  Offset is added back: xy stays full-frame pixels.
        x0 = int(CROP[0] * w)
        crop = frame[:, x0:int(CROP[1] * w)]
        # Detect everyone (cheap), pick the pitcher's box, and run the expensive pose net on that
        # one box only: pose-on-everyone cost ~1 s/frame with the crowd in view.
        boxes = _model.det_model(crop)   # (P, 4) x1,y1,x2,y2 in crop pixels
        if len(boxes) == 0:
            xy.append(np.zeros((17, 2))); conf.append(np.zeros(17)); continue
        full = boxes + [x0, 0, x0, 0]
        i = _pick_pitcher(full, w, h) if prev is None else _nearest(full, prev, w)
        prev = full[i]
        kp, sc = _model.pose_model(crop, bboxes=boxes[i:i + 1])
        xy.append(kp[0] + [x0, 0]); conf.append(sc[0])
    cap.release()
    return Pose2D(np.array(xy), np.array(conf), float(fps), (w, h))
