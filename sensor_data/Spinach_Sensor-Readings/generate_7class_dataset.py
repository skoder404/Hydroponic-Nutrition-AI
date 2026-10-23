#!/usr/bin/env python3
"""
Hybrid 7-class hydroponic spinach dataset.

REAL  : air temperature and relative humidity, sampled from the greenhouse
        loggers in Meng, Biradar, Park & Kelley (2025), Ag Data Commons
        (file: ..._greenhouse_temperature_humidity.csv).
SYNTH : nutrient-solution pH, EC, TDS and water temperature (not available in
        the real deposit). Water temperature follows the real air temperature;
        pH / EC are drawn inside or outside the healthy bands depending on the class.
LABELS: rule-based, from the 7 failure classes:

  Normal                     all variables optimal
  Water_Temp_High            water temperature > 21 C
  EC_High_Toxicity           EC > 2.3 mS/cm
  EC_Low_Deficiency          EC < 1.8 mS/cm
  pH_Imbalance               pH < 6.0 or pH > 7.0
  Humidity_High_Mildew_Risk  relative humidity > 75 %
  Humidity_Low_Stress        relative humidity < 45 %

Each non-Normal sample breaks exactly ONE rule; everything else stays in band.

CAVEAT: the real humidity never exceeds 74.05 %, so no real reading can be
Humidity_High. For that class the real rows with the highest humidity are
shifted above 75 % and marked humidity_source = "adjusted".

Usage:
  python3 generate_7class_dataset.py
  python3 generate_7class_dataset.py --per-class 300 --seed 7 --out-prefix big
"""
import argparse

import numpy as np
import pandas as pd

LABELS = [
    "Normal",
    "Water_Temp_High",
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "pH_Imbalance",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
]
# scarce classes are drawn first so they get first pick of the real rows
DRAW_ORDER = [
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "Water_Temp_High",
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "pH_Imbalance",
    "Normal",
]


def rule_label(df):
    """Re-derive the label from the sensor values (also used as a self-check)."""
    v = pd.DataFrame(
        {
            "Water_Temp_High": df.water_temperature_C > 21,
            "EC_High_Toxicity": df.ec_mS_cm > 2.3,
            "EC_Low_Deficiency": df.ec_mS_cm < 1.8,
            "pH_Imbalance": (df.ph < 6.0) | (df.ph > 7.0),
            "Humidity_High_Mildew_Risk": df.relative_humidity_percent > 75,
            "Humidity_Low_Stress": df.relative_humidity_percent < 45,
        }
    )
    n = v.sum(axis=1)
    return pd.Series(np.where(n == 0, "Normal", np.where(n == 1, v.idxmax(axis=1), "Multiple")), index=df.index)


def load_real(path, rng, test_frac):
    th = pd.read_csv(path)
    th["air"] = th.air_temperature_C.round(2)
    th["rh"] = th.relative_humidity_percent.round(2)
    ts = pd.to_datetime(th.timestamp_EDT)
    # 6-hour blocks per bench: whole blocks go to train OR test, so neighbouring
    # 10-minute readings never end up on both sides of the split
    th["block"] = th.bench + "|" + ts.dt.strftime("%Y-%m-%d") + "|" + (ts.dt.hour // 6).astype(str)
    blocks = th.block.unique()
    rng.shuffle(blocks)
    size = th.block.value_counts()
    test_blocks, acc = set(), 0
    for b in blocks:
        if acc >= test_frac * len(th):
            break
        test_blocks.add(b)
        acc += size[b]
    th["split"] = np.where(th.block.isin(test_blocks), "test", "train")
    return th


def draw(th, mask, split, n, used, rng, weights=None):
    cand = th.index[mask & (th.split == split) & ~th.index.isin(used)].to_numpy()
    replace = len(cand) < n
    if replace:
        print(f"  WARNING: only {len(cand)} unused real rows for {n} wanted ({split}); reusing rows")
        cand = th.index[mask & (th.split == split)].to_numpy()
    p = None
    if weights is not None:
        p = weights[cand] / weights[cand].sum()
    idx = rng.choice(cand, size=n, replace=replace, p=p)
    used.update(idx.tolist())
    return idx


def clipped(rng, mean, sd, lo, hi, n):
    return np.clip(rng.normal(mean, sd, n), lo, hi)


def make_chemistry(label, air, rng):
    n = len(air)
    # reservoir temperature follows the real air temperature, damped
    wt = 0.55 * air + 8.5 + rng.normal(0, 0.5, n)
    if label == "Water_Temp_High":
        wt = np.maximum(wt, 21.2) + rng.uniform(0.1, 3.0, n)
    else:
        wt = np.where(wt > 20.9, 20.9 - rng.uniform(0.1, 1.2, n), wt)

    ph = clipped(rng, 6.5, 0.25, 6.05, 6.95, n)
    if label == "pH_Imbalance":
        low = rng.random(n) < 0.5
        ph_low = np.clip(5.95 - np.abs(rng.normal(0, 0.55, n)), 4.2, 5.95)
        ph_high = np.clip(7.05 + np.abs(rng.normal(0, 0.55, n)), 7.05, 8.6)
        ph = np.where(low, ph_low, ph_high)

    ec = clipped(rng, 2.05, 0.12, 1.82, 2.28, n)
    if label == "EC_High_Toxicity":
        ec = np.clip(2.33 + np.abs(rng.normal(0, 0.35, n)), 2.33, 3.6)
    elif label == "EC_Low_Deficiency":
        ec = np.clip(1.77 - np.abs(rng.normal(0, 0.30, n)), 0.5, 1.77)

    tds = ec * 1000 * 0.7 * (1 + rng.normal(0, 0.02, n))
    return wt.round(2), ph.round(2), ec.round(2), tds.round(0).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-csv", default="Meng_Biradar_Park_Kelley_2025_greenhouse_temperature_humidity.csv")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="spinach_7class")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    th = load_real(args.real_csv, rng, args.test_frac)
    n_test = round(args.per_class * args.test_frac)
    n_train = args.per_class - n_test
    n_for = {"train": n_train, "test": n_test}
    used = set()
    frames = []

    # Humidity_High pool: real rows with the highest humidity. Lower the cutoff
    # until both splits have enough rows.
    cutoff = 70.0
    while cutoff > 50:
        m = th.rh >= cutoff
        if all((m & (th.split == s)).sum() >= n_for[s] for s in n_for):
            break
        cutoff -= 1
    print(f"Humidity_High source pool: real rows with RH >= {cutoff:.0f} % (shifted above 75 %)")

    for label in DRAW_ORDER:
        for split in ("train", "test"):
            n = n_for[split]
            if label == "Humidity_High_Mildew_Risk":
                mask = th.rh >= cutoff
            elif label == "Humidity_Low_Stress":
                mask = th.rh < 45
            else:
                mask = th.rh.between(45, 75)
            weights = np.exp(0.3 * th.air.to_numpy()) if label == "Water_Temp_High" else None
            idx = draw(th, mask, split, n, used, rng, weights)
            sub = th.loc[idx].copy()
            wt, ph, ec, tds = make_chemistry(label, sub.air.to_numpy(), rng)

            rh = sub.rh.to_numpy().copy()
            source = np.array(["real"] * n, dtype=object)
            if label == "Humidity_High_Mildew_Risk":
                pct = pd.Series(rh).rank(pct=True).to_numpy()
                rh = (75.5 + 9.0 * (0.7 * pct + 0.3 * rng.random(n))).round(2)
                source[:] = "adjusted"

            frames.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "source_timestamp_EDT": sub.timestamp_EDT.to_numpy(),
                        "bench": sub.bench.to_numpy(),
                        "air_temperature_C": sub.air.to_numpy(),
                        "relative_humidity_percent": rh,
                        "water_temperature_C": wt,
                        "ph": ph,
                        "ec_mS_cm": ec,
                        "tds_ppm": tds,
                        "label": label,
                        "humidity_source": source,
                    }
                )
            )

    df = pd.concat(frames, ignore_index=True)
    df["label_id"] = df.label.map({l: i for i, l in enumerate(LABELS)})

    # self-check: the rules must reproduce every label, with exactly one rule broken
    check = rule_label(df)
    bad = (check != df.label).sum()
    assert bad == 0, f"{bad} rows disagree with the labelling rules"

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df.insert(0, "sample_id", [f"S{i + 1:04d}" for i in range(len(df))])
    cols = ["sample_id", "air_temperature_C", "relative_humidity_percent", "water_temperature_C", "ph",
            "ec_mS_cm", "tds_ppm", "label", "label_id", "humidity_source", "source_timestamp_EDT", "bench", "split"]
    df = df[cols]

    out = args.out_prefix
    df.drop(columns="split").to_csv(f"{out}_full.csv", index=False)
    df[df.split == "train"].drop(columns="split").to_csv(f"{out}_train.csv", index=False)
    df[df.split == "test"].drop(columns="split").to_csv(f"{out}_test.csv", index=False)
    print(f"\nWrote {out}_full.csv ({len(df)}), {out}_train.csv ({(df.split == 'train').sum()}), {out}_test.csv ({(df.split == 'test').sum()})")
    print(pd.crosstab(df.label, df.split).loc[LABELS].to_string())


if __name__ == "__main__":
    main()
