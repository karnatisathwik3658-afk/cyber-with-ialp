
import pandas as pd
import numpy as np
import joblib
import time

dataset_path = r".\DataSet\kddcup.data.cleaned.txt"
encoder_path = r".\models\ordinal_encoder.pkl"
model_path = r".\models\isolation_forest_normal.pkl"

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
    "dst_host_srv_rerror_rate", "dst_host_srv_rerror_rate"
]

# Correct final column list
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

print("Loading encoder and model...")

encoder = joblib.load(encoder_path)
model = joblib.load(model_path)

print("Starting full-dataset evaluation...")
start_time = time.time()

tp = 0
tn = 0
fp = 0
fn = 0

total_records = 0
total_attacks = 0
total_normal = 0
total_anomalies = 0

chunk_number = 0

for chunk in pd.read_csv(
    dataset_path,
    names=columns,
    chunksize=100000
):
    chunk_number += 1

    actual_binary = (
        chunk["attack_type"].str.strip() != "normal."
    ).astype(int).to_numpy()

    X = chunk.drop(columns=["attack_type"])

    X[categorical_columns] = encoder.transform(
        X[categorical_columns]
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    predictions = model.predict(X)

    predicted_binary = (
        predictions == -1
    ).astype(int)

    tp += np.sum(
        (actual_binary == 1) &
        (predicted_binary == 1)
    )

    tn += np.sum(
        (actual_binary == 0) &
        (predicted_binary == 0)
    )

    fp += np.sum(
        (actual_binary == 0) &
        (predicted_binary == 1)
    )

    fn += np.sum(
        (actual_binary == 1) &
        (predicted_binary == 0)
    )

    total_records += len(chunk)
    total_attacks += np.sum(actual_binary == 1)
    total_normal += np.sum(actual_binary == 0)
    total_anomalies += np.sum(predicted_binary == 1)

    if chunk_number % 5 == 0:
        print(
            f"Processed approximately "
            f"{total_records:,} records..."
        )

accuracy = (tp + tn) / total_records

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0 else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0 else 0
)

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else 0
)

elapsed = time.time() - start_time

print("\n========== FULL DATASET EVALUATION ==========")
print("Total records:", f"{total_records:,}")
print("Actual attacks:", f"{total_attacks:,}")
print("Actual normal:", f"{total_normal:,}")
print("Predicted anomalies:", f"{total_anomalies:,}")

print("\nConfusion Matrix:")
print("True Positives:", f"{tp:,}")
print("True Negatives:", f"{tn:,}")
print("False Positives:", f"{fp:,}")
print("False Negatives:", f"{fn:,}")

print("\nMetrics:")
print("Accuracy:", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Recall:", round(recall, 6))
print("F1-Score:", round(f1_score, 6))
print("Specificity:", round(specificity, 6))

print("\nEvaluation time:", round(elapsed / 60, 2), "minutes")
print("\nFULL DATASET EVALUATION COMPLETED")