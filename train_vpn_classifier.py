
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a VPN traffic classifier on flow-level CSVs.

- Recursively loads CSVs from --data-dir
- Labels rows by:
    1) preferred label column (e.g., --label-column label)
    2) else 'is_vpn'/'vpn'/'target' column if present
    3) else filename inference using robust regex (checks NON-VPN before VPN)
- Drops identifier/leaky columns and engineered labels from features
- Trains RandomForest and MLP, selects best by F1
- Saves artifacts: model.pkl, metrics.json, columns.json, confusion_matrix.png, feature_importance.png, train_log.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Constants / defaults
# ----------------------------

# Drop IDs/high-cardinality and any column that can leak the label
DROP_COLS_DEFAULT = [
    "flow_id", "protocol", "a_ip", "b_ip", "a_port", "b_port",
    "is_vpn", "label", "application"  # prevent leakage
]

NONVPN_PAT = re.compile(r"\b(non[-_ ]?vpn|normal|benign|no[_-]?vpn)\b", re.I)
VPN_PAT    = re.compile(r"\b(vpn)\b", re.I)

# ----------------------------
# Data loading & labeling
# ----------------------------

def discover_csvs(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])


def infer_label_from_path(p: Path) -> int | None:
    lower = p.as_posix().lower()
    # Check NON-VPN first to avoid 'vpn' matching inside 'nonvpn'
    if NONVPN_PAT.search(lower):
        return 0
    if VPN_PAT.search(lower):
        return 1
    return None


def load_data(data_dir: Path, label_column: str | None) -> pd.DataFrame:
    csvs = discover_csvs(data_dir)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_dir.resolve()}")

    frames = []
    used_strategy = None

    for p in csvs:
        df = pd.read_csv(p)

        # Strategy 1: explicit label column provided by user
        if label_column and label_column in df.columns:
            df = df.copy()
            df["label"] = df[label_column].astype(int)
            used_strategy = f"label_column:{label_column}"
            frames.append(df)
            continue

        # Strategy 2: common label-like column names
        for cand in ["is_vpn", "vpn", "target"]:
            if cand in df.columns:
                df = df.copy()
                df["label"] = df[cand].astype(int)
                used_strategy = f"column:{cand}"
                frames.append(df)
                break
        else:
            # Strategy 3: filename inference
            inferred = infer_label_from_path(p)
            if inferred is None:
                # skip unlabeled file
                continue
            df = df.copy()
            df["label"] = int(inferred)
            used_strategy = "path_inference"
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "Could not assemble a labeled dataset. Ensure your CSVs contain a label "
            "column (e.g., is_vpn) OR are named with *_vpn.csv / *_nonvpn.csv."
        )

    data = pd.concat(frames, ignore_index=True)

    # Normalize dtypes: convert anything numeric-like to numeric if possible
    for c in data.columns:
        if c != "label":
            data[c] = pd.to_numeric(data[c], errors="ignore")

    print(f"[INFO] Loaded {len(data):,} rows from {len(frames)} files. Labeling: {used_strategy}")
    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Coerce to numeric where possible
    for c in out.columns:
        if c != "label":
            out[c] = pd.to_numeric(out[c], errors="ignore")

    # Compute robust ratios / rates if source columns exist
    def safe_div(n, d):
        return n / d.replace(0, np.nan)

    if {"packets_a_to_b", "packets_b_to_a"}.issubset(out.columns):
        out["packet_ratio"] = safe_div(out["packets_a_to_b"], out["packets_b_to_a"] + 1)

    if {"bytes_a_to_b", "bytes_b_to_a"}.issubset(out.columns):
        out["byte_ratio"] = safe_div(out["bytes_a_to_b"], out["bytes_b_to_a"] + 1)

    if {"total_packets", "duration"}.issubset(out.columns):
        out["packet_rate"] = out["total_packets"] / out["duration"].replace(0, np.nan)

    if {"total_bytes", "duration"}.issubset(out.columns):
        out["byte_rate"] = out["total_bytes"] / out["duration"].replace(0, np.nan)

    # Replace inf with NaN
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def select_features(df: pd.DataFrame, drop_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column (0=non-VPN, 1=VPN).")

    y = df["label"].astype(int)
    X = df.drop(columns=["label"] + [c for c in drop_cols if c in df.columns], errors="ignore")

    # Keep only numeric columns
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols]

    # Guard: ensure we have features
    if X.shape[1] == 0:
        raise RuntimeError("No numeric features left after dropping columns. Check your CSV schema.")

    return X, y, num_cols

# ----------------------------
# Models
# ----------------------------

def build_models(n_features: int) -> dict[str, BaseEstimator]:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )

    # RF pipeline (impute missing)
    rf_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", rf),
    ])

    mlp = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            random_state=42,
            verbose=False,
        ))
    ])

    return {"RandomForest": rf_pipe, "MLP": mlp}

# ----------------------------
# Evaluation & plotting
# ----------------------------

def evaluate(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    auc = None
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = None

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def plot_confusion(cm: np.ndarray, out_path: Path, labels=("Non-VPN", "VPN")):
    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=140)
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(model: Pipeline, feature_names: list[str], out_path: Path):
    # Only works for RF best model
    try:
        rf = model.named_steps["clf"]
        if not hasattr(rf, "feature_importances_"):
            return
        importances = rf.feature_importances_
        idx = np.argsort(importances)[::-1][:25]
        fig = plt.figure(figsize=(8, 8), dpi=140)
        plt.barh(range(len(idx)), importances[idx][::-1])
        plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1])
        plt.xlabel("Importance")
        plt.title("Feature Importance (RandomForest)")
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception:
        # silently skip if structure differs
        pass

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VPN traffic classifier on flow-level CSVs.")
    parser.add_argument("--data-dir", type=str, default="datasets/csvs", help="Directory containing CSVs (recursively).")
    parser.add_argument("--label-column", type=str, default=None, help="Optional: name of label column (0/1).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--drop-cols", type=str, nargs="*", default=DROP_COLS_DEFAULT,
                        help=f"Columns to drop. Default: {DROP_COLS_DEFAULT}")
    parser.add_argument("--out-dir", type=str, default="artifacts",
                        help="Directory to save model and reports.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "train_log.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            data = load_data(Path(args.data_dir), args.label_column)
            data = engineer_features(data)
            X, y, used_cols = select_features(data, args.drop_cols)

            # Guard: must have both classes
            if len(np.unique(y)) < 2:
                raise RuntimeError(
                    "Only one class found in labels. "
                    "Ensure both VPN and Non-VPN samples are present "
                    "(e.g., *_vpn.csv and *_nonvpn.csv)."
                )

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state,
                stratify=y
            )

            models = build_models(X_train.shape[1])
            results = {}
            best_name = None
            best_f1 = -1.0
            best_model = None

            for name, model in models.items():
                model.fit(X_train, y_train)
                metrics = evaluate(model, X_test, y_test)
                results[name] = metrics
                log.write(f"== {name} ==\n{metrics['classification_report']}\n")
                log.flush()
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_name = name
                    best_model = model

            # Save artifacts
            with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"results": results, "best_model": best_name}, f, indent=2)

            joblib.dump(best_model, out_dir / "model.pkl")

            with open(out_dir / "columns.json", "w", encoding="utf-8") as f:
                json.dump({"feature_columns": used_cols}, f, indent=2)

            cm = np.array(results[best_name]["confusion_matrix"])
            plot_confusion(cm, out_dir / "confusion_matrix.png")

            if best_name == "RandomForest":
                plot_feature_importance(best_model, used_cols, out_dir / "feature_importance.png")

            print(f"[OK] Best model: {best_name}  F1={best_f1:.4f}")
            print(f"Artifacts saved to {out_dir.resolve()}")
            log.write(f"\n[OK] Best model: {best_name}  F1={best_f1:.4f}\n")

        except Exception as e:
            traceback.print_exc()
            log.write("\n[ERROR]\n" + traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    main()
