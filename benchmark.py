#!/usr/bin/env python3
"""
Credit Card Fraud Detection - LightGBM Benchmark Script
Designed for Lab 16: Cloud AI Environment Setup (GCP / AWS / OCI)

This script:
1. Loads the Credit Card Fraud dataset (or synthetic data if requested/missing).
2. Splits dataset into stratified train, validation, and test sets.
3. Trains a LightGBM Classifier with early stopping.
4. Evaluates test set performance: AUC-ROC, Accuracy, F1-Score, Precision, Recall.
5. Measures inference latency (1 row) and inference throughput (1000 rows).
6. Outputs metrics to console and writes benchmark_result.json.
"""

import os
import sys
import time
import json
import argparse
import platform

# Ensure UTF-8 output encoding across different OS and terminal settings
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import lightgbm as lgb


def find_dataset(custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    candidates = [
        "creditcard.csv",
        os.path.expanduser("~/ml-benchmark/creditcard.csv"),
        "./ml-benchmark/creditcard.csv",
        "../creditcard.csv",
        "/home/creditcard.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def generate_synthetic_data(num_samples=10000):
    print("Generating synthetic Credit Card dataset for verification...")
    np.random.seed(42)
    # Features: Time, V1..V28, Amount
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    data = np.random.randn(num_samples, len(cols))
    df = pd.DataFrame(data, columns=cols)
    df["Time"] = np.sort(np.random.uniform(0, 172800, size=num_samples))
    df["Amount"] = np.abs(np.random.exponential(scale=88, size=num_samples))
    # Highly imbalanced: ~0.17% fraud
    fraud_prob = 0.0017
    df["Class"] = np.random.choice([0, 1], size=num_samples, p=[1 - fraud_prob, fraud_prob])
    if df["Class"].sum() == 0:
        df.loc[np.random.choice(num_samples, size=15, replace=False), "Class"] = 1
    return df


def run_benchmark(data_path=None, output_json="benchmark_result.json", demo_mode=False):
    print("=" * 65)
    print("      LAB 16: CLOUD AI BENCHMARK - LIGHTGBM ON CPU")
    print("=" * 65)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()} | LightGBM: {lgb.__version__}")
    print("-" * 65)

    # 1. Load Data
    t0_load = time.perf_counter()
    if demo_mode:
        print("[1/5] Demo mode active: using synthetic data...")
        df = generate_synthetic_data(num_samples=20000)
        data_source = "synthetic (demo)"
    else:
        resolved_path = find_dataset(data_path)
        if resolved_path is None:
            print("[!] Dataset 'creditcard.csv' not found in standard locations.")
            print("    Please download it using Kaggle CLI:")
            print("      mkdir -p ~/ml-benchmark")
            print("      kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/")
            print("    Or pass --demo to run with synthetic data for testing.")
            sys.exit(1)
        
        print(f"[1/5] Loading dataset from: {resolved_path}")
        df = pd.read_csv(resolved_path)
        data_source = resolved_path

    load_time = time.perf_counter() - t0_load
    print(f"      Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns in {load_time:.3f}s")
    
    if "Class" not in df.columns:
        print("[!] Target column 'Class' not found in dataset!")
        sys.exit(1)

    fraud_count = int(df["Class"].sum())
    normal_count = int(len(df) - fraud_count)
    fraud_pct = (fraud_count / len(df)) * 100
    print(f"      Distribution: {normal_count:,} Normal (0), {fraud_count:,} Fraud (1) [{fraud_pct:.2f}%]")

    # 2. Train/Test Split
    print("[2/5] Splitting dataset (Train 72%, Val 8%, Test 20%)...")
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified split: 80% train+val, 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # 10% of train_full for validation (8% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.10, random_state=42, stratify=y_train_full
    )

    # 3. Model Training
    print("[3/5] Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=15, verbose=False)
    ]

    t0_train = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=callbacks
    )
    train_time = time.perf_counter() - t0_train
    best_iter = getattr(model, "best_iteration_", model.n_estimators) or model.n_estimators
    print(f"      Training complete in {train_time:.3f}s (Best iteration: {best_iter})")

    # 4. Evaluation on Test Set
    print("[4/5] Evaluating model on test set...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))

    # 5. Latency & Throughput Benchmark
    print("[5/5] Measuring inference latency and throughput...")
    # Single row latency
    single_row = X_test.iloc[[0]]
    for _ in range(10):  # warmup
        _ = model.predict_proba(single_row)

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict_proba(single_row)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
    avg_latency_1row_ms = float(np.mean(latencies))

    # 1000 rows throughput
    batch_size = min(1000, len(X_test))
    batch_data = X_test.iloc[:batch_size]
    for _ in range(5):  # warmup
        _ = model.predict_proba(batch_data)

    batch_times = []
    for _ in range(25):
        t0 = time.perf_counter()
        _ = model.predict_proba(batch_data)
        batch_times.append(time.perf_counter() - t0)
    avg_batch_time = float(np.mean(batch_times))
    throughput_1000_rows = float(batch_size / avg_batch_time)

    # Summary Table
    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS TABLE")
    print("=" * 65)
    table_rows = [
        ("Thời gian load data", f"{load_time:.3f} s"),
        ("Thời gian training", f"{train_time:.3f} s"),
        ("Best iteration", f"{best_iter}"),
        ("AUC-ROC", f"{auc:.4f}"),
        ("Accuracy", f"{acc:.4f} ({acc*100:.2f}%)"),
        ("F1-Score", f"{f1:.4f}"),
        ("Precision", f"{prec:.4f}"),
        ("Recall", f"{rec:.4f}"),
        ("Inference latency (1 row)", f"{avg_latency_1row_ms:.3f} ms"),
        ("Inference throughput (1000 rows)", f"{throughput_1000_rows:.1f} rows/s"),
    ]

    print(f"| {'Metric':<34} | {'Kết quả':<24} |")
    print(f"|{'-'*36}|{'-'*26}|")
    for metric, result in table_rows:
        print(f"| {metric:<34} | {result:<24} |")
    print("=" * 65)

    # Save to JSON
    result_data = {
        "dataset_source": data_source,
        "total_samples": int(df.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "metrics": {
            "data_load_time_seconds": round(load_time, 4),
            "training_time_seconds": round(train_time, 4),
            "best_iteration": int(best_iter),
            "auc_roc": round(auc, 4),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "inference_latency_1_row_ms": round(avg_latency_1row_ms, 4),
            "inference_throughput_1000_rows_per_sec": round(throughput_1000_rows, 2)
        },
        "system_info": {
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "lightgbm_version": lgb.__version__
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n[+] Saved complete metrics to: {os.path.abspath(output_json)}")
    print(json.dumps(result_data, indent=2))
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Lab 16 LightGBM CPU Benchmark")
    parser.add_argument("--data-path", type=str, default=None, help="Path to creditcard.csv")
    parser.add_argument("--output", type=str, default="benchmark_result.json", help="Path to output JSON")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data for testing")
    args = parser.parse_args()

    run_benchmark(
        data_path=args.data_path,
        output_json=args.output,
        demo_mode=args.demo
    )


if __name__ == "__main__":
    main()
