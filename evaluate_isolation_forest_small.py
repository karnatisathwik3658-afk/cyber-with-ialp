
import pandas as pd
import joblib
import numpy as np

dataset_path = r".\DataSet\kddcup.data.cleaned.txt"
encoder_path = r".\models\ordinal_encoder.pkl"
model_path = r".\models\isolation_forest_model.pkl"

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

categorical_columns = ["protocol_type", "service", "flag"]

normal_records = []
attack_records = []

target_per_class = 5000

print("Collecting a balanced sample...")

for chunk in pd.read_csv(
    dataset_path,
    names=columns,
    chunksize=100000
):
    normal = chunk[chunk["attack_type"].str.strip() == "normal."]
    attacks = chunk[chunk["attack_type"].str.strip() != "normal."]

    if len(normal_records) < target_per_class:
        remaining = target_per_class - len(normal_records)
        normal_records.extend(
            normal.head(remaining).to_dict("records")
        )

    if len(attack_records) < target_per_class:
        remaining = target_per_class - len(attack_records)
        attack_records.extend(
            attacks.head(remaining).to_dict("records")
        )

    if (
        len(normal_records) >= target_per_class
        and len(attack_records) >= target_per_class
    ):
        break

data = pd.DataFrame(normal_records + attack_records)

print("Normal records:", len(normal_records))
print("Attack records:", len(attack_records))
print("Total sample:", len(data))

# Actual labels
actual_binary = (
    data["attack_type"].str.strip() != "normal."
).astype(int)

# Load model and encoder
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)

# Prepare features
X = data.drop(columns=["attack_type"])

X[categorical_columns] = encoder.transform(
    X[categorical_columns]
)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print("Generating predictions...")

predictions = model.predict(X)
predicted_binary = (predictions == -1).astype(int)

tp = np.sum((actual_binary == 1) & (predicted_binary == 1))
tn = np.sum((actual_binary == 0) & (predicted_binary == 0))
fp = np.sum((actual_binary == 0) & (predicted_binary == 1))
fn = np.sum((actual_binary == 1) & (predicted_binary == 0))

accuracy = (tp + tn) / len(actual_binary)
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

print("\n========== BALANCED EVALUATION ==========")
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

print("\nBALANCED EVALUATION COMPLETED")