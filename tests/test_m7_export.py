"""M7 Task 5: face export layout. Fixture is SYNTHETIC (scripts/m7_export.py --synthetic)."""
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m7_export import main  # noqa: E402

FIX = ROOT / "web" / "test" / "fixtures" / "face-synthetic.json"


def test_layout_and_label():
    d = json.load(open(FIX))
    assert d["meta"]["schema"] == "face-1" and d["meta"]["synthetic"] is True
    assert d["meta"]["label"] == "Simulated (M2 + M1/M3, picked by M6)"
    assert set(d["seq"]["classes"]) == set(d["bank"]) == set(d["types"]) == set(d["location"])
    assert set(d["hitter"]) == {"L", "R"} and set(d["zone"]) == {"L", "R"}
    assert all("foul_tip_share" in h for h in d["hitter"].values())


def test_rare_type_absent_from_bank_is_never_in_seq_classes():
    d = json.load(open(FIX))
    for e in d["excluded"]:
        assert e["type"] not in d["seq"]["classes"] and e["type"] not in d["bank"]


def test_parity_block():
    d = json.load(open(FIX))
    assert len(d["parity"]) == 20
    p = d["parity"][0]
    assert {"ctx", "type", "type_probs", "loc_mean", "hitter", "reaim"} <= set(p)
    assert abs(sum(p["type_probs"].values()) - 1) < 1e-9


def test_synthetic_may_not_leave_fixtures(tmp_path):
    with pytest.raises(SystemExit):
        main(["--synthetic", "data/synthetic.csv", "--out", str(tmp_path / "x.json")])


def test_size_budget():   # real exports, skipped if not yet written
    files = list((ROOT / "web/public/data/face").glob("*.json"))
    if not files:
        pytest.skip("no real face exports yet")
    for f in files:
        assert len(gzip.compress(f.read_bytes())) <= 1_000_000
