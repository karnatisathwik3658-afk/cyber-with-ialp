
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

dataset_path = r".\DataSet\kddcup.data.cleaned.txt"
encoder_path = r".\models\ordinal_encoder.pkl"
rf_path = r".\models\random_forest_model.pkl"
xgb_path = r".\models\xgboost_model.json"

columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type"
]

categorical_columns = [
    "protocol_type", "service", "flag"
]

print("Loading models...")

encoder = joblib.load(encoder_path)
rf_model = joblib.load(rf_path)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(xgb_path)

print("Models loaded successfully.")

# Confusion-matrix totals
results = {
    "Random Forest": [0, 0, 0, 0],
    "XGBoost": [0, 0, 0, 0]
}

total_records = 0
start_time = time.time()

print("Starting full-dataset evaluation...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        dataset_path,
        names=columns,
        chunksize=100000
    ),
    start=1
):
    actual = (
        chunk["attack_type"].str.strip() != "normal."
    ).astype(int).to_numpy()

    X = chunk.drop(columns=["attack_type"])

    X[categorical_columns] = encoder.transform(
        X[categorical_columns]
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    predictions = {
        "Random Forest": rf_model.predict(X),
        "XGBoost": xgb_model.predict(X)
    }

    for model_name, predicted in predictions.items():
        predicted = np.asarray(predicted).astype(int).reshape(-1)

        # Handle possible XGBoost probability output
        if not np.all(np.isin(predicted, [0, 1])):
            predicted = (predicted >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        ).ravel()

        results[model_name][0] += tn
        results[model_name][1] += fp
        results[model_name][2] += fn
        results[model_name][3] += tp

    total_records += len(chunk)

    if chunk_number % 5 == 0:
        print(
            f"Processed approximately "
            f"{total_records:,} records..."
        )

print("\n========== MODEL COMPARISON ==========")
print("Total records:", f"{total_records:,}")

for model_name, values in results.items():
    tn, fp, fn, tp = values

    accuracy = (tp + tn) / total_records
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    specificity = (
        tn / (tn + fp)
        if tn + fp else 0
    )

    print(f"\n--- {model_name} ---")
    print("True Positives:", f"{tp:,}")
    print("True Negatives:", f"{tn:,}")
    print("False Positives:", f"{fp:,}")
    print("False Negatives:", f"{fn:,}")
    print("Accuracy:", round(accuracy, 6))
    print("Precision:", round(precision, 6))
    print("Recall:", round(recall, 6))
    print("F1-Score:", round(f1, 6))
    print("Specificity:", round(specificity, 6))

elapsed = time.time() - start_time
print("\nEvaluation time:", round(elapsed / 60, 2), "minutes")
print("\nMODEL COMPARISON COMPLETED")