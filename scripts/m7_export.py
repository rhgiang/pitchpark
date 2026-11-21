"""M7 "Face the pitcher" export: per-pitcher static JSON the browser engine reads (schema face-1).

    python scripts/m7_export.py --all                    # 8 pitchers -> web/public/data/face/<slug>.json
    python scripts/m7_export.py sale-chris
    python scripts/m7_export.py --synthetic --out web/test/fixtures/face-synthetic.json

Everything is fitted on split_by_game(seed=0) TRAINING games: M2 arm + M3 push per newest season
(seam.fit_pitch_types), M6 pitch type / location / average hitter on all seasons pooled. Each
pitcher gets a bank of aimed, pre-flown M2 pitches (face.make_bank). Before anything is written the
browser's cheap re-aim (face.reaim_knots) is checked against a true re-simulation; if any max error
exceeds 0.5 in (path or plate) the pitcher is NOT written and the script exits non-zero.
Writes results/m7/reaim.csv (local). Synthetic output may only go under web/test/fixtures/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import m6_evaluate
import m6_location
import m7_hitter
from pitchsim.data import RAW_DIR, add_air_density, fetch_game_conditions, load_seasons
from pitchsim.export import KNOT_S, write_json
from pitchsim.face import make_bank, reaim_error_in, reaim_knots
from pitchsim.generative import FEATURES as M2_FEATURES, MIN_TRAIN
from pitchsim.physics import PhysicsParams, air_density
from pitchsim.seam import fit_pitch_types
from pitchsim.sequence.context import add_context
from pitchsim.sequence.hitter import HALF_W_FT, HitterModel, HitterTable, prepare, zone_bin
from pitchsim.sequence.location import LocationModel
from pitchsim.sequence.logistic import FEATURES as SEQ_BASE, LogisticSeq
from pitchsim.statcast import clean
from pitchsim.synthetic import synthetic_at_bats
from pitchsim.validation import split_by_game

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "web" / "test" / "fixtures"
FACE_DIR = ROOT / "web" / "public" / "data" / "face"
REAIM_CSV = ROOT / "results" / "m7" / "reaim.csv"
SLUGS = {"sale-chris": (519242, "Sale"), "ohtani-shohei": (660271, "Ohtani"),
         "misiorowski-jacob": (694819, "Misiorowski"), "cease-dylan": (656302, "Cease"),
         "yamamoto-yoshinobu": (808967, "Yamamoto"), "king-michael": (650633, "King"),
         "sanchez-cristopher": (650911, "Sanchez"), "schlittler-cam": (693645, "Schlittler")}
SEQ_FEATURES = SEQ_BASE + ["tto"]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3]
FLOOR = 1e-4
MIN_LOC = 30          # located training pitches of one type
N_GATE = 50           # bank pitches per type in the re-aim gate
GATE_MAX_IN = 0.5
SEED = 0
LABEL = "Simulated (M2 + M1/M3, picked by M6)"


def _round(o, nd=4):
    """Floats rounded to nd dp, recursively (bank entries)."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def _cats(enc, features):
    return {f: [str(v) for v in c] for f, c in zip(features, enc.categories_)}


def _softmax_coef(lr):
    """(classes x features) coef and intercept; sklearn gives one row for two classes."""
    w, b = lr.coef_, lr.intercept_
    if len(lr.classes_) == 2:   # softmax(0, wx+b) == sigmoid
        w, b = np.vstack([np.zeros_like(w), w]), np.array([0.0, b[0]])
    return w, b


def load_real(slug: str):
    """(newest-season cleaned frame with rho, pooled context+location frame, season, raw)."""
    pid, _ = SLUGS[slug]
    raw = load_seasons(sorted(RAW_DIR.glob(f"{pid}_*.csv")))
    season = int(raw["game_year"].max())
    new = clean(raw[raw["game_year"] == season].reset_index(drop=True))
    rho = add_air_density(new, fetch_game_conditions(new["game_pk"].unique()))["rho"]
    pooled = m6_location.add_location(add_context(raw).dropna(subset=["pitch_type"]))
    return new.assign(rho=rho.to_numpy()), pooled, season, raw


def build(slug: str, synthetic: bool) -> tuple[dict, list[dict]]:
    """(face JSON dict, reaim gate rows) for one pitcher."""
    if synthetic:
        raw = synthetic_at_bats(seed=SEED)
        season = int(raw["game_year"].max())
        new = clean(raw.reset_index(drop=True)).assign(rho=air_density(70.0, 0.0))
        pooled = m6_location.add_location(add_context(raw).dropna(subset=["pitch_type"]))
        short = "synthetic"
    else:
        new, pooled, season, raw = load_real(slug)
        short = SLUGS[slug][1]

    # --- M2 + M3 on the newest season's training games
    train_new, _, fits, _ = fit_pitch_types(new, new["rho"].to_numpy())
    train_new = train_new.dropna(subset=M2_FEATURES)
    # --- M6 on pooled seasons' training games
    train, _ = split_by_game(pooled, seed=SEED)
    train_loc = train.dropna(subset=["x_ft", "z_ft"])

    all_types = sorted(set(pooled["pitch_type"]) | set(new["pitch_type"]))
    excluded, ok = [], []
    for t in all_types:
        n_m2 = int((train_new["pitch_type"] == t).sum())
        n_loc = int((train_loc["pitch_type"] == t).sum())
        if t not in fits:
            excluded.append({"type": t, "reason": "M3 has no fit (too few training or held-out pitches)"})
        elif n_m2 < MIN_TRAIN:
            excluded.append({"type": t, "reason": f"M2: {n_m2} training pitches (< {MIN_TRAIN})"})
        elif n_loc < MIN_LOC:
            excluded.append({"type": t, "reason": f"{n_loc} located training pitches (< {MIN_LOC})"})
        else:
            ok.append(t)
    if len(ok) < 2:
        raise SystemExit(f"{slug}: fewer than 2 modelled pitch types")
    types = sorted(ok)

    # --- location: ridge alpha tuned like m6_location, one model per type
    alpha = m6_location.tune(train_loc[train_loc["pitch_type"].isin(types)])
    locs = {t: LocationModel(alpha=alpha).fit(train_loc[train_loc["pitch_type"] == t]) for t in types}
    loc_means = {t: LocationModel(features=[]).fit(train_loc[train_loc["pitch_type"] == t]).mean(
        train_loc.iloc[:1])[0] for t in types}

    # --- aimed bank (M2 mixture arm, M1 + M3 flight)
    bank = _round(make_bank(train_new, train_new["rho"].to_numpy(), {t: fits[t] for t in types},
                            loc_means, n_per_type=200, seed=SEED))

    # --- re-aim gate: targets from each type's location model across the training contexts
    rng = np.random.default_rng(SEED)
    rho_med = float(np.median(train_new["rho"]))
    params = {t: PhysicsParams(cd=fits[t].cd, spin_efficiency=fits[t].eff, rho=rho_med) for t in types}
    sub, targets = {}, {}
    for t in types:
        sub[t] = bank[t][:N_GATE]
        rows = train_loc[train_loc["pitch_type"] == t]
        ctx = rows.iloc[rng.integers(0, len(rows), N_GATE)]
        targets[t] = locs[t].sample(ctx, rng)
    err = reaim_error_in(sub, targets, params)
    gate = []
    for t, g in err.groupby("pitch_type"):
        off = np.hypot(g["target_x_ft"] - loc_means[t][0], g["target_z_ft"] - loc_means[t][1])
        gate.append({"pitcher": slug, "type": t, "n": len(g),
                     "median_in": g.path_in.median(), "p95_in": g.path_in.quantile(0.95),
                     "max_in": g.path_in.max(), "plate_median_in": g.plate_in.median(),
                     "plate_p95_in": g.plate_in.quantile(0.95), "plate_max_in": g.plate_in.max(),
                     "max_offset_ft": off.max()})

    # --- sequencing: logistic + tto on the bank types only (so classes == bank types)
    train_seq = train[train["pitch_type"].isin(types)]
    make = lambda ts, h: LogisticSeq(ts, C=h, features=SEQ_FEATURES)
    C = m6_evaluate.tune(make, C_GRID, train_seq, types)
    seq = make(types, C).fit(train_seq)
    enc, lr = seq.model[0], seq.model[1]
    assert list(lr.classes_) == types
    w, b = _softmax_coef(lr)

    # --- hitter: model or table per stand (hitter.csv `use`), tuned and fitted on all training games
    c_h, a_h = m7_hitter.tune(train)
    hit_model, hit_table = HitterModel(c_h).fit(train), HitterTable(a_h).fit(train)
    use = {}
    if not synthetic:
        h = pd.read_csv(ROOT / "results" / "m7" / "hitter.csv")
        use = {r.stand: r.use for r in h[h.pitcher == short].itertuples()}
    prep = prepare(train)
    hitter = {}
    for s in ("L", "R"):
        kind = use.get(s, "model")
        j = (hit_model if kind == "model" else hit_table).to_json()
        w2 = prep[(prep["outcome"] == "whiff") & (prep["strikes"] == 2) & (prep["stand"] == s)]
        j["foul_tip_share"] = float((w2["description"] == "foul_tip").mean()) if len(w2) else 0.0
        hitter[s] = {"kind": kind, **j}
    zone = {s: {"top_ft": float(train[train["stand"] == s]["sz_top"].median()),
                "bot_ft": float(train[train["stand"] == s]["sz_bot"].median())} for s in ("L", "R")}

    names = new.dropna(subset=["pitch_type"]).groupby("pitch_type")["pitch_name"].first()
    arm = train_new["arm_angle"] if "arm_angle" in train_new else None

    def arm_deg(t):
        if arm is None:
            return None
        v = arm[train_new["pitch_type"] == t].dropna()
        return float(v.median()) if len(v) else None

    out = {
        "meta": {"schema": "face-1", "synthetic": synthetic, "label": LABEL, "seed": SEED,
                 "knot_s": KNOT_S, "season": season, "train_games": int(train_new["game_pk"].nunique())},
        "pitcher": {"name": str(raw["player_name"].iloc[0]), "id": int(raw["pitcher"].iloc[0]),
                    "throws": str(raw["p_throws"].iloc[0])},
        "types": {t: {"name": str(names.get(t, t)), "arm_angle_deg": arm_deg(t),
                      "n_train": int((train_new["pitch_type"] == t).sum())} for t in types},
        "excluded": excluded,
        "seq": {"features": SEQ_FEATURES, "categories": _cats(enc, SEQ_FEATURES), "classes": types,
                "coef": w.tolist(), "intercept": b.tolist(), "floor": FLOOR, "C": C},
        "location": {t: {"features": locs[t].features, "categories": _cats(locs[t].enc, locs[t].features),
                         "coef": locs[t].ridge.coef_.tolist(), "intercept": locs[t].ridge.intercept_.tolist(),
                         "cov": locs[t].cov.tolist(), "aim": [float(v) for v in loc_means[t]]} for t in types},
        "hitter": hitter, "zone": zone, "bank": {t: bank[t] for t in types}}
    if synthetic:
        out["parity"] = parity(out, seq, locs, hit_model if hitter["L"]["kind"] == "model" else hit_table)
    return out, gate


def parity(out: dict, seq: LogisticSeq, locs: dict, hit) -> list[dict]:
    """20 contexts with Python's exact outputs for the JS port (synthetic fixture only).

    zone_bin uses np.digitize(right=True): a distance exactly on an edge goes to the LOWER bin;
    two spots sit on the plate edge so the JS port can match that.
    """
    rng = np.random.default_rng(SEED)
    cat = out["seq"]["categories"]
    types = out["seq"]["classes"]
    spots = [(0.0, 2.5), (0.4, 3.0), (-0.9, 1.4), (0.2, 4.2), (HALF_W_FT, 2.5), (HALF_W_FT + 0.25, 2.5)]
    rows = []
    for i in range(20):
        ctx = {"count": str(rng.choice(cat["count"])), "stand": "LR"[i % 2],
               "prevType": str(rng.choice(cat["prev_type"])), "tto": str(rng.choice(cat["tto"]))}
        if i in (3, 11):
            ctx["prevType"] = "ZZ"     # unseen level: handle_unknown="ignore"
        if i == 7:
            ctx["tto"] = "9"
        if i == 15:
            ctx["count"] = "9-9"
        t = types[i % len(types)]
        df = pd.DataFrame([{"count": ctx["count"], "stand": ctx["stand"], "prev_type": ctx["prevType"],
                            "tto": ctx["tto"]}])
        probs = seq.predict(df)[0]     # floor + renormalise exactly as LogisticSeq.predict
        x, z = spots[i % len(spots)]
        zn = out["zone"][ctx["stand"]]
        hd = pd.DataFrame([{"count": ctx["count"], "stand": ctx["stand"], "pitch_type": t,
                            "x_ft": x, "z_ft": z, "sz_top": zn["top_ft"], "sz_bot": zn["bot_ft"]}])
        zb = str(zone_bin(hd["x_ft"], hd["z_ft"], hd["sz_top"], hd["sz_bot"]).iloc[0])
        hp = hit.predict(hd.assign(zone_bin=zb))[0]
        dx, dz = 0.6, -0.4
        rows.append({"ctx": ctx, "type": t, "type_probs": dict(zip(types, probs.tolist())),
                     "loc_mean": locs[t].mean(df)[0].tolist(),
                     "hitter": {"x": x, "z": z, "sz_top": zn["top_ft"], "sz_bot": zn["bot_ft"],
                                "zone_bin": zb, "probs": hp.tolist()},
                     "reaim": {"type": t, "index": 0, "dx": dx, "dz": dz,
                               "knots": reaim_knots(out["bank"][t][0]["sim"], dx, dz)}})
    return rows


def write_gate(rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    REAIM_CSV.parent.mkdir(parents=True, exist_ok=True)
    if REAIM_CSV.exists():
        old = pd.read_csv(REAIM_CSV)
        new = pd.concat([old[~old.pitcher.isin(new.pitcher)], new], ignore_index=True)
    new.round(4).to_csv(REAIM_CSV, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--synthetic", nargs="?", const=True, default=False,
                    help="SYNTHETIC fixture (synthetic.synthetic_at_bats; any value is ignored)")
    ap.add_argument("--out", help="output dir (real) or file (--synthetic, under web/test/fixtures)")
    a = ap.parse_args(argv)

    if a.synthetic:
        out = Path(a.out or FIXTURES / "face-synthetic.json").resolve()
        if FIXTURES.resolve() not in out.parents:
            raise SystemExit(f"Synthetic data may only be written to {FIXTURES}")
        data, gate = build("synthetic", True)
        g = pd.DataFrame(gate)
        print(g.round(3).to_string(index=False))
        if max(g.max_in.max(), g.plate_max_in.max()) > GATE_MAX_IN:
            raise SystemExit("BLOCKED: synthetic re-aim gate failed")
        write_json(data, out)
        print(f"wrote {out}")
        return 0

    slugs = list(SLUGS) if a.all else a.slugs
    unknown = [s for s in slugs if s not in SLUGS]
    if not slugs or unknown:
        raise SystemExit(f"give slugs from {list(SLUGS)} or --all (unknown: {unknown})")
    out_dir = Path(a.out) if a.out else FACE_DIR
    failed = []
    for slug in slugs:
        data, gate = build(slug, False)
        write_gate(gate)
        g = pd.DataFrame(gate)
        print(f"\n{slug}\n{g.round(3).to_string(index=False)}", flush=True)
        if max(g.max_in.max(), g.plate_max_in.max()) > GATE_MAX_IN:
            failed.append(slug)       # BLOCKED: no file for this pitcher
            continue
        write_json(data, out_dir / f"{slug}.json")
        print(f"wrote {out_dir / slug}.json C={data['seq']['C']} excluded={[e['type'] for e in data['excluded']]}",
              flush=True)
    if failed:
        print(f"BLOCKED: re-aim gate failed for {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
