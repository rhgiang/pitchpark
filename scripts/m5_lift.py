"""3D lift of every screened-ok clip with HMR 2.0 (4D-Humans).  Runs ONLY in .venv-pose (torch).
usage: .venv-pose/Scripts/python.exe scripts/m5_lift.py data/video/ohtani-shohei/pose2d.csv [clip ...]

Why HMR 2.0 and not GVHMR/WHAM: see results/M5.md ("Spike and clips").  It is a per-frame model: each frame is
cropped around the pitcher and a ViT regresses SMPL pose + shape + a weak-perspective camera.  The
pitcher is chosen by our own RTMPose tracking (Task 2): the crop box is the bounding box of his 2D
keypoints, so the lifter cannot jump to the batter or catcher.
Output: data/video/<slug>/lift/<clip>.npz (see pitchsim.pose.joints3d), joints in the CAMERA frame
(x right, y down, z away from camera, metres), with `up` estimated from the body (below).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HMR2_ROOT = REPO / "third_party" / "4D-Humans"
os.environ["HMR2_CACHE_PARENT"] = str(REPO / "third_party" / "hmr2_cache_parent")  # -> <it>/4DHumans
# The checkpoint is the official 4D-Humans one (pickled omegaconf objects); torch>=2.6 refuses those by
# default.  Fine to allow for this one trusted file.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
sys.path[:0] = [str(HMR2_ROOT), str(REPO / "src")]

import types

import cv2
import numpy as np
import pandas as pd
import torch

from pitchsim.pose.joints3d import Lift3D, save_lift
from pitchsim.pose.pose2d import load_pose2d

# hmr2 imports pyrender for its demo renderer; we never render, and pyrender needs EGL (absent on
# Windows), so give hmr2 an empty stand-in module.
_pyr = types.ModuleType("pyrender")
def _stub(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return type(name, (), {})


_pyr.__getattr__ = _stub
sys.modules.setdefault("pyrender", _pyr)

# chumpy (unpickles the SMPL .pkl) predates Python 3.11 / NumPy 1.24: restore what it still uses.
import inspect

inspect.getargspec = inspect.getfullargspec
for _n, _t in dict(bool=bool, int=int, float=float, complex=complex, object=object, str=str, unicode=str).items():
    if not hasattr(np, _n):
        setattr(np, _n, _t)

SEED = 0
CONF_MIN = 0.3   # keypoint confidence needed to count towards the person's box
PAD = 0.25       # box padding: RTMPose keypoints are joint centres, the body extends beyond them
LIFTER = "hmr2.0a-per-frame"
OHTANI_HEIGHT_M = 1.93


def person_box(xy: np.ndarray, conf: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """xyxy box around confident keypoints; reuse the previous box if the frame has none."""
    k = conf > CONF_MIN
    if k.sum() < 4:
        return prev if prev is not None else np.array([0, 0, 1, 1], np.float32)
    p = xy[k]
    lo, hi = p.min(0), p.max(0)
    pad = PAD * (hi - lo)
    return np.concatenate([lo - pad, hi + pad]).astype(np.float32)


def estimate_up(j: np.ndarray, fps: float) -> np.ndarray:
    """Gravity-up in the lifter's frame from the body itself (a per-frame model has no world frame).

    Why: the pitcher stands in the set position for the first part of the clip, so the vector from
    the ankle midpoint to the neck over the first ~25% of frames points along gravity up (to within
    the few degrees he leans).  Averaging over the whole clip would mix in the delivery's lean."""
    n = max(5, len(j) // 4)
    v = j[:n, 12] - 0.5 * (j[:n, 7] + j[:n, 8])        # neck - ankle midpoint (SMPL idx 12, 7, 8)
    u = np.median(v, axis=0)
    return (u / np.linalg.norm(u)).astype(np.float64)


@torch.no_grad()
def lift_clip(model, cfg, smpl_neutral, video: Path, p2d, device) -> tuple[Lift3D, dict]:
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    n = min(len(frames), len(p2d.xy))
    boxes, prev = [], None
    for i in range(n):
        prev = person_box(p2d.xy[i], p2d.conf[i], prev)
        boxes.append(prev)
    boxes = np.stack(boxes)
    h, w = frames[0].shape[:2]
    joints, betas, heights = [], [], []
    for i in range(n):
        item = ViTDetDataset(cfg, frames[i], boxes[i:i + 1])[0]
        batch = {k: (torch.as_tensor(v)[None].to(device)) for k, v in item.items() if k != "personid"}
        out = model(batch)
        cam = out["pred_cam"]
        size = batch["img_size"]
        # crop-camera -> full-image camera translation (the same formula as 4D-Humans' demo)
        focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * size.max()
        bs = batch["box_size"] * cam[:, 0] + 1e-9
        t = torch.stack([2 * (batch["box_center"][:, 0] - size[:, 0] / 2) / bs + cam[:, 1],
                         2 * (batch["box_center"][:, 1] - size[:, 1] / 2) / bs + cam[:, 2],
                         2 * focal / bs], dim=-1)
        P = out["pred_smpl_params"]
        so = smpl_neutral(global_orient=P["global_orient"], body_pose=P["body_pose"], betas=P["betas"],
                          pose2rot=False)
        joints.append((so.joints[0, :24] + t[0]).cpu().numpy())   # first 24 = SMPL joints, in order
        betas.append(P["betas"][0].cpu().numpy())
        heights.append(float((so.vertices[0, :, 1].max() - so.vertices[0, :, 1].min()).cpu()))
    j = np.stack(joints).astype(np.float32)
    up = estimate_up(j, p2d.fps)
    # Body height: the clip's median shape (betas) posed in the rest pose, so crouching in the
    # delivery doesn't shrink it.  Measured along y (the rest pose's up).
    b = torch.as_tensor(np.median(np.stack(betas), 0))[None].to(device)
    z = torch.eye(3, device=device).expand(1, 1, 3, 3)
    rest = smpl_neutral(global_orient=z, body_pose=torch.eye(3, device=device).expand(1, 23, 3, 3), betas=b,
                        pose2rot=False)
    rest_h = float((rest.vertices[0, :, 1].max() - rest.vertices[0, :, 1].min()).cpu())
    info = dict(rest_height_m=rest_h, median_frame_extent_m=float(np.median(heights)))
    return Lift3D(j, p2d.fps, up, LIFTER), info


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    from hmr2.models import HMR2, load_hmr2, DEFAULT_CHECKPOINT
    import smplx
    csv = Path(sys.argv[1])
    only = set(sys.argv[2:])
    df = pd.read_csv(csv)
    df = df[df["ok"] == True]
    if only:
        df = df[df["clip"].isin(only)]
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    # Our own SMPL layer, not model.smpl: hmr2's wrapper remaps joints to OpenPose order; a plain
    # SMPLLayer gives the 24 SMPL joints first, in SMPL order (what web/src/skeleton.js expects).
    smpl_neutral = smplx.SMPLLayer(model_path=str(REPO / "third_party/hmr2_cache_parent/4DHumans/data/smpl"),
                                   gender="neutral").to(device).eval()
    outdir = csv.parent / "lift"
    import time
    for clip in df["clip"]:
        dst = outdir / clip.replace(".mp4", ".npz")
        if dst.exists():
            continue
        t0 = time.time()
        p2d = load_pose2d(csv.parent / "pose2d" / clip.replace(".mp4", ".npz"))
        lift, info = lift_clip(model, cfg, smpl_neutral, csv.parent / clip, p2d, device)
        save_lift(lift, dst)
        print(f"{clip}: {len(lift.joints_m)} frames, {time.time() - t0:.1f}s, "
              f"body height (median shape, rest pose) {info['rest_height_m']:.2f} m "
              f"(Ohtani {OHTANI_HEIGHT_M} m), median per-frame extent {info['median_frame_extent_m']:.2f} m, "
              f"up={np.round(lift.up, 2)}", flush=True)


if __name__ == "__main__":
    main()
