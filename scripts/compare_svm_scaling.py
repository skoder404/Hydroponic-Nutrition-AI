"""Compare SVM performance with and without scaling on Lettuce sensor data (T4-013)."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.evaluation.metrics import calculate_metrics

if __name__ == "__main__":
    tr = pd.read_csv("processed_data/sensor/lettuce/train.csv")
    val = pd.read_csv("processed_data/sensor/lettuce/val.csv")
    feats = ["ph", "ec", "water_temp", "humidity", "air_temp"]
    labels = sorted(tr["label"].unique())

    X_tr, y_tr = tr[feats], tr["label"].values
    X_v, y_v = val[feats], val["label"].values

    # 1. Scaled SVM Pipeline
    pipe_scaled = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
        ]
    )
    pipe_scaled.fit(X_tr, y_tr)
    y_pred_scaled = pipe_scaled.predict(X_v)
    m_scaled = calculate_metrics(y_v, y_pred_scaled, labels=labels)

    # 2. Unscaled SVM Pipeline
    pipe_unscaled = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
        ]
    )
    pipe_unscaled.fit(X_tr, y_tr)
    y_pred_unscaled = pipe_unscaled.predict(X_v)
    m_unscaled = calculate_metrics(y_v, y_pred_unscaled, labels=labels)

    print("--- SCALING COMPARISON RESULTS ---")
    print(
        f"With Scaling:    Macro-F1 = {m_scaled['macro_f1']:.4f} | Accuracy = {m_scaled['accuracy']:.4f}"
    )
    print(
        f"Without Scaling: Macro-F1 = {m_unscaled['macro_f1']:.4f} | Accuracy = {m_unscaled['accuracy']:.4f}"
    )
    diff = m_scaled["macro_f1"] - m_unscaled["macro_f1"]
    print(f"Macro-F1 Gain:   {diff:+.4f}")
