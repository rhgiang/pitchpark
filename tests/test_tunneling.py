import numpy as np

from pitchsim.tunneling import (PERCEIVED_REF_EXT_FT, approach_angles_deg, perceived_velocity_mph,
                                position_at_y, separation_in)


def _path(dx=0.0):
    t = np.linspace(0, 0.4, 5)
    pos = np.column_stack([np.full(5, dx), np.linspace(55, 1.4, 5), np.linspace(6, 2, 5)])
    return t, pos


def test_vaa_flat_and_descending():
    v = np.array([[0, -130.0, 0], [0, -130.0, -130 * np.tan(np.radians(6))]])
    vaa, haa = approach_angles_deg(v)
    assert abs(vaa[0]) < 1e-9 and abs(vaa[1] + 6) < 1e-9
    assert np.allclose(haa, 0)


def test_haa_sign():
    assert approach_angles_deg(np.array([[5.0, -130.0, 0]]))[1][0] > 0


def test_perceived_equals_speed_at_reference():
    assert np.isclose(perceived_velocity_mph(90.0, PERCEIVED_REF_EXT_FT), 90.0)


def test_perceived_increases_with_extension():
    assert perceived_velocity_mph(90.0, 7.0) > perceived_velocity_mph(90.0, 6.0) > perceived_velocity_mph(90.0, 5.0)


def test_identical_paths_zero_separation():
    assert separation_in(_path(), _path()) == 0.0


def test_offset_path_known_separation():
    assert np.isclose(separation_in(_path(), _path(0.25)), 3.0)


def test_position_at_y_between_samples():
    t = np.array([0.0, 1.0])
    pos = np.array([[0.0, 50.0, 6.0], [2.0, 0.0, 2.0]])
    assert np.allclose(position_at_y(t, pos, 25.0), [1.0, 4.0])
