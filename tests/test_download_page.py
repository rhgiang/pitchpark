import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("dl", Path(__file__).parents[1] / "scripts" / "m5_download_page.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def test_exclude_home_drops_those_games(tmp_path):
    rows = [(g, 1, 1, f"p{g}", "train", "FF") for g in (1, 2, 3)]
    pd.DataFrame(rows, columns=["game_pk", "at_bat_number", "pitch_number", "play_id", "split", "pitch_type"]
                 ).to_csv(tmp_path / "clips.csv", index=False)
    home = pd.Series({1: "NYY", 2: "BOS", 3: "TB"})
    sel = dl.choose(tmp_path, 3, 0, home=home, exclude_home=["NYY"])
    assert set(sel["game_pk"]) == {2, 3}
