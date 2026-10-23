#!/usr/bin/env python3
"""
Build a balanced, labelled hydroponic-tomato training set from the Kolar
land-based tomato sensor file.

Run on your FULL file:
  python build_tomato_dataset.py kolar_tomato_precision_agri_dataset.csv

What it does
  1. Keeps only the columns that matter (drops RSSI, soil-transport physics, soil moisture,
     Nutrient_Lockout_Prob, ...). Timestamp is used only to split train/test.
  2. Maps land-based (soil) values onto hydroponic nutrient-solution scales (see MAPPING below).
  3. Generates the missing water temperature from air temperature + solar radiation.
  4. Labels every row with 1 of 11 classes using hydroponic-tomato reference ranges.
  5. Balances the classes (equal rows per class) and splits train/test by calendar-month
     blocks, so no month appears on both sides.

MAPPING (proxy transforms, state them in your report)
  air_temperature_C          = Ambient_Temp                      (unchanged)
  relative_humidity_percent  = Humidity + 15      (a greenhouse is more humid than outside air)
  ph                         = Soil_pH - 1.0      (soil 6.2-8.5 -> solution 5.2-7.5)
  ec_mS_cm                   = EC_mS_cm x 1.4     (soil 0.5-3.0 -> solution 0.7-4.2)
  nitrogen_mg_L              = Nitrogen_Level x 4   (10-50  -> 40-200 mg/L)
  phosphorus_mg_L            = Phosphorus_Level x 3 (5-25   -> 15-75 mg/L)
  potassium_mg_L             = Potassium_Level x 6  (15-60  -> 90-360 mg/L)
  water_temperature_C        = SYNTHETIC: 8 + 0.55 x air temp + 0.003 x (solar - 700) + noise

CLASSES
  Default mode "primary": a row that breaks no rule is Normal. A row that breaks one or more
  rules gets the class of its MOST SEVERE fault, where severity = how far past the threshold the
  value is, as a fraction of the largest such deviation in the file (0 = at the threshold, 1 = worst
  value seen). 10 classes. This uses the data well (thousands of rows per class on 2 lakh rows).
  Mode "single" (--mode single): only rows breaking exactly ONE rule keep that class; rows breaking
  two or more become an extra Multiple_Faults class. Cleaner labels, but far fewer usable rows.

  Normal                     no rule broken
  Water_Temp_High            water temperature > 25 C
  EC_High_Toxicity           EC > 4.0 mS/cm
  EC_Low_Deficiency          EC < 2.0 mS/cm
  pH_Imbalance               pH < 6.0 or pH > 6.5
  Humidity_High_Mildew_Risk  humidity > 85 %
  Humidity_Low_Stress        humidity < 50 %
  N_Deficiency               nitrogen < 90 mg/L
  P_Deficiency               phosphorus < 35 mg/L
  K_Deficiency               potassium < 144 mg/L
  Multiple_Faults            (single mode only) two or more of the above

Reference ranges: Ohio State CFAES / UA-CEA tomato solution (N 90-190, P 47, K 144-350 mg/L,
pH 5.5-6.5, EC about 2.0-2.4), Oklahoma State extension (tomato pH 6.0-6.5, EC 2.0-4.0),
Hort Americas (optimum tomato root temperature 25 C), Baptista et al. 2008 / Virginia Tech
(humidity: low stress below 50 %, leaf mould risk above 85 %).
"""
import argparse

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- settings
PH_SHIFT = -1.0
EC_SCALE = 1.4
HUM_SHIFT = 15.0
N_SCALE, P_SCALE, K_SCALE = 4.0, 3.0, 6.0

T = dict(water_high=25.0, ec_low=2.0, ec_high=4.0, ph_low=6.0, ph_high=6.5,
         hum_low=50.0, hum_high=85.0, n_low=90.0, p_low=35.0, k_low=144.0)

SINGLE = ["Water_Temp_High", "EC_High_Toxicity", "EC_Low_Deficiency", "pH_Imbalance",
          "Humidity_High_Mildew_Risk", "Humidity_Low_Stress", "N_Deficiency", "P_Deficiency", "K_Deficiency"]
NEEDED = ["Timestamp", "Ambient_Temp", "Humidity", "Solar_Radiation", "Soil_pH", "EC_mS_cm",
          "Nitrogen_Level", "Phosphorus_Level", "Potassium_Level"]
FEATURES = ["air_temperature_C", "relative_humidity_percent", "water_temperature_C", "ph", "ec_mS_cm",
            "nitrogen_mg_L", "phosphorus_mg_L", "potassium_mg_L"]


def deviations(df):
    """How far each value is past its healthy limit (positive = rule broken)."""
    return pd.DataFrame({
        "Water_Temp_High": df.water_temperature_C - T["water_high"],
        "EC_High_Toxicity": df.ec_mS_cm - T["ec_high"],
        "EC_Low_Deficiency": T["ec_low"] - df.ec_mS_cm,
        "pH_Imbalance": np.maximum(T["ph_low"] - df.ph, df.ph - T["ph_high"]),
        "Humidity_High_Mildew_Risk": df.relative_humidity_percent - T["hum_high"],
        "Humidity_Low_Stress": T["hum_low"] - df.relative_humidity_percent,
        "N_Deficiency": T["n_low"] - df.nitrogen_mg_L,
        "P_Deficiency": T["p_low"] - df.phosphorus_mg_L,
        "K_Deficiency": T["k_low"] - df.potassium_mg_L,
    })


def assign_labels(df, mode, scales):
    dev = deviations(df)
    viol = dev > 0
    n = viol.sum(axis=1)
    if mode == "single":
        multi = np.where(n == 1, viol.idxmax(axis=1), "Multiple_Faults")
        lab = np.where(n == 0, "Normal", multi)
    else:  # primary: most severe fault, severity scaled to 0-1 per rule
        severity = dev.clip(lower=0) / scales
        lab = np.where(n == 0, "Normal", severity.idxmax(axis=1))
    return pd.Series(lab, index=df.index), n


def map_to_hydroponic(raw, rng):
    out = pd.DataFrame({
        "air_temperature_C": raw.Ambient_Temp.round(2),
        "relative_humidity_percent": (raw.Humidity + HUM_SHIFT).round(2),
        "water_temperature_C": (8.0 + 0.55 * raw.Ambient_Temp + 0.003 * (raw.Solar_Radiation - 700)
                                + rng.normal(0, 0.6, len(raw))).round(2),
        "ph": (raw.Soil_pH + PH_SHIFT).round(2),
        "ec_mS_cm": (raw.EC_mS_cm * EC_SCALE).round(2),
        "nitrogen_mg_L": (raw.Nitrogen_Level * N_SCALE).round(1),
        "phosphorus_mg_L": (raw.Phosphorus_Level * P_SCALE).round(1),
        "potassium_mg_L": (raw.Potassium_Level * K_SCALE).round(1),
    }, index=raw.index)
    out["source_row"] = raw.index.to_numpy()
    out["source_timestamp"] = raw.Timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="the big Kolar CSV")
    ap.add_argument("--per-class", type=int, default=5000, help="max rows per class (default 5000; fewer if a class is rare)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["primary", "single"], default="primary",
                    help="primary = label by most severe fault (default); single = strict one-fault rows + Multiple_Faults")
    ap.add_argument("--out-prefix", default="tomato_hydroponic")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = pd.read_csv(args.csv, usecols=NEEDED, parse_dates=["Timestamp"])
    print(f"Read {len(raw):,} rows")
    df = map_to_hydroponic(raw, rng)
    scales = deviations(df).clip(lower=0).max().clip(lower=1e-9)  # worst deviation per rule, for severity
    df["label"], df["n_rules_broken"] = assign_labels(df, args.mode, scales)

    classes = ["Normal"] + SINGLE + (["Multiple_Faults"] if args.mode == "single" else [])
    df = df[df.label.isin(classes)].copy()

    # split by calendar month, so no month appears in both train and test
    month = pd.to_datetime(df.source_timestamp).dt.strftime("%Y-%m")
    months = month.unique()
    rng.shuffle(months)
    size = month.value_counts()
    test_months, acc = set(), 0
    for m in months:
        if acc >= args.test_frac * len(df):
            break
        test_months.add(m)
        acc += size[m]
    df["split"] = np.where(month.isin(test_months), "test", "train")

    pools = pd.crosstab(df.label, df.split).reindex(classes)
    print("\nAvailable rows per class (after mapping):")
    print(pools.to_string())
    tf = args.test_frac
    avail = min(int(min(pools.loc[c, "train"] / (1 - tf), pools.loc[c, "test"] / tf)) for c in classes)
    per_class = min(args.per_class, avail)
    n_test = round(per_class * tf)
    n_train = per_class - n_test
    print(f"\nRows per class: {per_class} (train {n_train}, test {n_test}); limited by the rarest class = {avail}")

    parts = []
    for c in classes:
        for split, n in (("train", n_train), ("test", n_test)):
            pool = df[(df.label == c) & (df.split == split)]
            parts.append(pool.sample(n=n, random_state=args.seed))
    out = pd.concat(parts)

    # self-check: re-derive every label from the saved values
    assert (assign_labels(out, args.mode, scales)[0] == out.label).all(), "label mismatch"

    out["label_id"] = out.label.map({c: i for i, c in enumerate(classes)})
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out.insert(0, "sample_id", [f"T{i + 1:05d}" for i in range(len(out))])
    cols = ["sample_id"] + FEATURES + ["label", "label_id", "n_rules_broken", "source_row", "source_timestamp"]

    p = args.out_prefix
    out[cols].to_csv(f"{p}_full.csv", index=False)
    out[out.split == "train"][cols].to_csv(f"{p}_train.csv", index=False)
    out[out.split == "test"][cols].to_csv(f"{p}_test.csv", index=False)
    print(f"\nWrote {p}_full.csv ({len(out):,}), {p}_train.csv ({(out.split == 'train').sum():,}), {p}_test.csv ({(out.split == 'test').sum():,})")
    print(pd.crosstab(out.label, out.split).reindex(classes).to_string())


if __name__ == "__main__":
    main()
