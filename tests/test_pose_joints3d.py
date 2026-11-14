import numpy as np
import pytest

from pitchsim.pose.joints3d import Lift3D, load_lift, save_lift


def _synthetic_lift(n_joints=24) -> Lift3D:
    """SYNTHETIC data (random walk of joints), only to test file I/O."""
    rng = np.random.default_rng(0)
    return Lift3D(
        joints_m=rng.normal(size=(10, n_joints, 3)).astype(np.float32),
        fps=29.97,
        up=np.array([0.0, 1.0, 0.0]),
        lifter="synthetic-test",
    )


def test_load_lift_roundtrip(tmp_path):
    lift = _synthetic_lift()
    save_lift(lift, tmp_path / "a" / "clip.npz")
    got = load_lift(tmp_path / "a" / "clip.npz")
    assert got.joints_m.shape == (10, 24, 3) and got.joints_m.dtype == np.float32
    assert got.fps == pytest.approx(29.97)
    assert np.allclose(got.up, [0, 1, 0])
    assert got.lifter == "synthetic-test"
    assert np.array_equal(got.joints_m, lift.joints_m)


def test_load_lift_rejects_wrong_joint_count(tmp_path):
    save_lift(_synthetic_lift(n_joints=17), tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="24"):
        load_lift(tmp_path / "bad.npz")
