"""Tests for the fetch/cache/conditions layer.  No network."""

import numpy as np
import pandas as pd
import pytest

from pitchsim.data import (VENUE_ELEVATION_FT, add_air_density, fetch_game_conditions,
                           parse_game_feed)
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
