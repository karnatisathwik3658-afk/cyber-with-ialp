
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

categorical_columns = [
    "protocol_type", "service", "flag"
]

print("Loading model and encoder...")

encoder = joblib.load(encoder_path)
model = joblib.load(model_path)

print("Reading dataset sample...")

data = pd.read_csv(
    dataset_path,
    names=columns,
    nrows=10000
)

# Actual labels
actual_labels = data["attack_type"].str.strip()

# Normal = 0, Attack = 1
actual_binary = (
    actual_labels != "normal."
).astype(int)

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

# Isolation Forest:
# 1 = normal-like
# -1 = anomaly-like
predicted_binary = (predictions == -1).astype(int)

tp = np.sum((actual_binary == 1) & (predicted_binary == 1))
tn = np.sum((actual_binary == 0) & (predicted_binary == 0))
fp = np.sum((actual_binary == 0) & (predicted_binary == 1))
fn = np.sum((actual_binary == 1) & (predicted_binary == 0))

accuracy = (tp + tn) / len(actual_binary)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\n========== EVALUATION RESULTS ==========")
print("Total records:", len(X))
print("Actual attacks:", np.sum(actual_binary == 1))
print("Actual normal:", np.sum(actual_binary == 0))

print("\nConfusion Matrix:")
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

print("\nMetrics:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

print("\nEVALUATION COMPLETED")