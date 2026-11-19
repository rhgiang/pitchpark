"""M4: export pitchers' held-out at-bats as JSON for the batter's-eye viewer.

    python scripts/export_viewer.py data/raw/<pitcher>.csv [more.csv ...] [--out web/public/data]

Writes <out>/<slug>.json per pitcher (a pitcher's season CSVs are pooled) and <out>/index.json = [{slug, name, throws, n_at_bats, height_in, mix}].
CSV paths are explicit (no globbing data/raw/). Synthetic pitchers may only be written to
web/test/fixtures/ so nothing fake lands next to real pitchers in the viewer.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from pitchsim.data import add_air_density, fetch_game_conditions, fetch_heights, load_seasons
from pitchsim.export import export_pitcher, write_json
from pitchsim.physics import air_density

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "web" / "test" / "fixtures"


def counts_by_type(data: dict) -> dict:
    """Pitch count per pitch type over the export's pitches (what the menu's mix bar shows)."""
    types = pd.Series([p["pitch_type"] for a in data["at_bats"] for p in a["pitches"]]).dropna()
    return {k: int(v) for k, v in types.value_counts().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "web" / "public" / "data"))
    ap.add_argument("--no-weather", action="store_true",
                    help="use a constant air density (synthetic data)")
    ap.add_argument("--dt", type=float, default=0.005, help="path sampling step (s)")
    args = ap.parse_args()
    out = Path(args.out)

    index = []
    for _, raw in load_seasons(args.csv).groupby("player_name", sort=False):  # seasons pooled per pitcher
        raw = raw.reset_index(drop=True)
        name = str(raw["player_name"].iloc[0])
        # Strip accents: the slug regex would turn an accented letter into a hyphen.
        plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        if "Synthetic" in name and FIXTURES.resolve() not in [out.resolve(), *out.resolve().parents]:
            raise SystemExit(f"Synthetic data may only be written to {FIXTURES}")
        slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")

        if args.no_weather:
            rho = pd.Series(air_density(70.0, 0.0), index=raw.index)
        else:
            rho = add_air_density(raw, fetch_game_conditions(raw["game_pk"].unique()))["rho"]
        # Synthetic pitchers have no StatsAPI record.
        h = float("nan") if "Synthetic" in name else fetch_heights([int(raw["pitcher"].iloc[0])]).iloc[0]
        height = None if pd.isna(h) else int(h)  # unknown height: the viewer falls back to its default
        data = export_pitcher(raw, rho, dt_s=args.dt, height_in=height)
        path = out / f"{slug}.json"
        write_json(data, path)
        counts = pd.Series([p["status"] for a in data["at_bats"] for p in a["pitches"]])
        print(f"{plain}: {len(data['at_bats'])} at-bats, "
              f"{counts.value_counts().to_dict()}, {path.stat().st_size / 1e6:.2f} MB -> {path}")
        # height and pitch counts let the start menu draw its cards without downloading every pitcher
        index.append({"slug": slug, "name": name, "throws": data["pitcher"]["throws"],
                      "n_at_bats": len(data["at_bats"]), "height_in": height,
                      "mix": counts_by_type(data)})

    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"wrote {out / 'index.json'}")


if __name__ == "__main__":
    main()
