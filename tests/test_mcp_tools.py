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


def test_compare_diffs_shared_types_only():
    other = {"pitcher": {"name": "Other"}, "at_bats": [{"id": "9-1", "pitches": [
        _p("FF", 97), _p("CU", 78)]}]}
    r = T.compare(DATA, other)
    assert r["shared"]["FF"]["median_mph"] == {"a": 95.0, "b": 97.0, "diff": 2.0}   # a: median(95, 95, 96)
    assert r["only_a"] == ["CH", "SL"] and r["only_b"] == ["CU"]
    assert r["shared"].keys() == {"FF"}


def test_sequence_transitions_filters_and_start_row():
    def ab(stand, *ps):
        return {"stand": stand, "pitches": [dict(_p(t, 90, sep=s), balls=b, strikes=k) for t, s, b, k in ps]}
    d = {"at_bats": [ab("R", ("FF", None, 0, 0), ("SL", 2.0, 0, 1), ("FF", 9.0, 0, 2)),
                     ab("R", ("FF", None, 0, 0), ("SL", 4.0, 0, 1)),
                     ab("L", ("FF", None, 0, 0), ("CH", 6.0, 0, 1))]}
    r = T.sequence(d)
    assert r["start"]["n"] == 3 and r["start"]["next"]["FF"]["p"] == 1.0
    assert r["FF"]["n"] == 3 and r["FF"]["next"]["SL"] == {"n": 2, "p": 0.667, "median_sep_in": 3.0}
    assert T.sequence(d, stand="L")["FF"]["next"].keys() == {"CH"}
    assert T.sequence(d, strikes=2) == {"SL": {"n": 1, "next": {"FF": {"n": 1, "p": 1.0, "median_sep_in": 9.0}}}}


def test_generator_check_filters_and_rejects_bad_input(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "summary.csv").write_text(
        "pitch_type,model,auc,auc_gap,pass\nFF,gaussian,0.66,0.01,True\nSL,gaussian,,,\n")
    assert T.generator_check("x", "FF", tmp_path) == [
        {"pitch_type": "FF", "model": "gaussian", "auc": 0.66, "auc_gap": 0.01, "pass": True}]
    assert T.generator_check("x", "SL", tmp_path)[0]["auc"] is None     # NaN becomes null
    with pytest.raises(FileNotFoundError):
        T.generator_check("nobody", None, tmp_path)
    with pytest.raises(ValueError):
        T.generator_check("../x", None, tmp_path)


def test_sequence_model_check_matches_surname_and_flags_unevaluated(tmp_path):
    f = tmp_path / "s.json"
    f.write_text('{"Sale": {"n_pitches": 10}}')
    assert T.sequence_model_check("sale-chris", f) == {"evaluated": True, "n_pitches": 10}
    assert T.sequence_model_check("king-michael", f)["evaluated"] is False   # not an error


def test_next_pitch_probs_validates_before_fitting():
    with pytest.raises(ValueError):
        T.next_pitch_probs("sale-chris", 4, 0, "R")       # 4 balls
    with pytest.raises(ValueError):
        T.next_pitch_probs("../x", 0, 0, "R")             # bad slug


def test_next_pitch_probs_sums_to_one_and_samples_reproducibly():
    r = T.next_pitch_probs("sale-chris", 0, 2, "R", "FF", 2, sample=True, seed=3)
    assert abs(sum(r["probs"].values()) - 1) < 0.01 and r["sampled"] in r["probs"]
    assert r["sampled"] == T.next_pitch_probs("sale-chris", 0, 2, "R", "FF", 2, sample=True, seed=3)["sampled"]
