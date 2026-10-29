"""Tests for the fetch/cache/conditions layer.  No network."""

import json

import numpy as np
import pandas as pd
import pytest

from pitchsim.data import (VENUE_ELEVATION_FT, add_air_density, fetch_game_conditions,
                           fetch_pitcher, lookup_id, parse_game_feed)
from pitchsim.physics import air_density

COORS = {"gamePk": 1, "gameData": {"venue": {"id": 19, "name": "Coors Field"},
                                   "weather": {"temp": "72", "condition": "Clear"}}}


def test_parse_game_feed():
    assert parse_game_feed(COORS) == {"game_pk": 1, "venue_id": 19,
                                      "venue_name": "Coors Field", "temp_f": 72.0}


def test_parse_game_feed_missing_weather():
    feed = {"gamePk": 2, "gameData": {"venue": {"id": 19, "name": "Coors Field"}}}
    assert np.isnan(parse_game_feed(feed)["temp_f"])
    feed["gameData"]["weather"] = {"temp": ""}
    assert np.isnan(parse_game_feed(feed)["temp_f"])


def test_add_air_density_uses_venue_and_temp():
    pitches = pd.DataFrame({"game_pk": [1, 2, 3]})
    cond = pd.DataFrame({"game_pk": [1, 2, 3], "venue_id": [19, 19, 999999],
                         "venue_name": ["Coors Field", "Coors Field", "Nowhere"],
                         "temp_f": [72.0, np.nan, 60.0]})
    out = add_air_density(pitches, cond)
    coors_ft = VENUE_ELEVATION_FT[19][1]
    assert out["rho"].tolist() == pytest.approx([air_density(72, coors_ft),
                                                 air_density(70, coors_ft),
                                                 air_density(60, 0)])
    assert "rho" not in pitches


def test_coors_is_high():
    assert 5150 < VENUE_ELEVATION_FT[19][1] < 5250


def test_fetch_game_conditions_uses_cache(tmp_path, monkeypatch):
    cache = tmp_path / "gc.csv"
    pd.DataFrame([parse_game_feed(COORS)]).to_csv(cache, index=False)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("network hit"))
    assert fetch_game_conditions([1], cache).loc[0, "venue_id"] == 19


def _feed(pk):
    return {"gamePk": pk, "gameData": {"venue": {"id": 19, "name": "Coors Field"},
                                       "weather": {"temp": "70"}}}


class _Resp:
    def __init__(self, feed):
        self.feed = feed

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return json.dumps(self.feed).encode()


def test_fetch_game_conditions_resumes_after_failure(tmp_path, monkeypatch):
    cache = tmp_path / "gc.csv"

    def flaky(url, timeout):
        if "/10/" in url:
            return _Resp(_feed(10))
        raise OSError("timeout")

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    with pytest.raises(OSError):
        fetch_game_conditions([10, 20], cache)
    assert pd.read_csv(cache)["game_pk"].tolist() == [10]

    asked = []

    def record(url, timeout):
        asked.append(url)
        return _Resp(_feed(20))

    monkeypatch.setattr("urllib.request.urlopen", record)
    out = fetch_game_conditions([10, 20], cache)
    assert len(asked) == 1 and "/20/" in asked[0]
    assert out["game_pk"].tolist() == [10, 20]


def test_fetch_pitcher_empty_result_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("pitchsim.data.RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr("pybaseball.statcast_pitcher", lambda *a, **k: pd.DataFrame())
    with pytest.raises(SystemExit):
        fetch_pitcher(None, 1, "2025-04-01", "2025-04-02")
    assert not list(tmp_path.rglob("*.csv"))


def test_lookup_id_no_match(monkeypatch):
    monkeypatch.setattr("pybaseball.playerid_lookup",
                        lambda last, first: pd.DataFrame(columns=["key_mlbam"]))
    with pytest.raises(SystemExit, match="No MLB player found"):
        lookup_id("Nobody Here")
