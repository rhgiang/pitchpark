"""MCP tool functions: hand-built viewer JSON (synthetic, tiny) with answers worked out by hand."""
import json
from pathlib import Path

import pytest

from pitchsim import mcp_tools as T

ROOT = Path(__file__).resolve().parents[1]


def _p(t, mph, sep=None, real=(0.0, 2.0, 0.4), sim=(0.0, 2.0, 0.4), status="ok", bx=-6.0):
    return {"pitch_type": t, "status": status, "speed_mph": mph, "spin_rpm": 2000.0,
            "plate_real": list(real), "plate_sim": list(sim) if status == "ok" else None,
            "tunnel": {"vaa_deg": -5.0, "haa_deg": 0.0, "perceived_mph": mph + 1,
                       "sep_prev_in": sep, "break_x_in": bx, "break_z_in": 10.0}}


# FF -> SL (sep 2.0), FF -> SL (sep 4.0), SL -> FF (sep 9.0), FF -> CH (no tracking: sep None)
DATA = {"meta": {"synthetic": True}, "pitcher": {"name": "Synthetic Test"}, "at_bats": [
    {"id": "1-1", "pitches": [_p("FF", 95), _p("SL", 85, sep=2.0), _p("FF", 95, sep=9.0)]},
    {"id": "1-2", "pitches": [_p("FF", 96), _p("SL", 84, sep=4.0, real=(0.0, 2.0, 0.4),
                                             sim=(0.0 + 3 / 12, 2.0 + 4 / 12, 0.4)),
                              _p("CH", 88, sep=None)]}]}


def test_tunnel_uses_consecutive_pitches_only():
    r = T.tunnel(DATA, "FF", "SL")
    assert r["n"] == 2 and r["median_sep_in"] == 3.0     # median(2, 4); the SL->FF 9.0 is excluded
    assert T.tunnel(DATA, "FF", "CH")["n"] == 0          # untracked pitch: no separation, no crash


def test_plate_error_is_distance_in_inches():
    r = T.plate_error(DATA)
    assert r["SL"]["median_miss_in"] == pytest.approx((0 + 5.0) / 2)   # 3-4-5 triangle: 5 in; other is 0
    assert r["FF"]["median_miss_in"] == 0.0


def test_arsenal_groups_by_type():
    a = T.arsenal(DATA)
    assert a["FF"]["n"] == 3 and a["SL"]["median_mph"] == 84.5


def test_load_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        T.load("../secrets", tmp_path)


def test_runs_on_synthetic_fixture():
    d = json.loads((ROOT / "web" / "test" / "fixtures" / "synthetic.json").read_text())
    assert d["meta"]["synthetic"] and T.arsenal(d) and T.plate_error(d)
