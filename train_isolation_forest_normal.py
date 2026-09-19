
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

dataset_path = r".\DataSet\kddcup.data.cleaned.txt"
encoder_path = r".\models\ordinal_encoder.pkl"
output_path = r".\models\isolation_forest_normal.pkl"

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

target_records = 200000
normal_records = []

print("Collecting normal traffic...")

for chunk in pd.read_csv(
    dataset_path,
    names=columns,
    chunksize=100000
):
    normal = chunk[
        chunk["attack_type"].str.strip() == "normal."
    ]

    remaining = target_records - len(normal_records)

    if remaining > 0:
        normal_records.extend(
            normal.head(remaining).to_dict("records")
        )

    if len(normal_records) >= target_records:
        break

data = pd.DataFrame(normal_records)

print("Normal records collected:", len(data))

# Remove labels
X = data.drop(columns=["attack_type"])

# Load encoder
encoder = joblib.load(encoder_path)

# Encode categorical features
X[categorical_columns] = encoder.transform(
    X[categorical_columns]
)

# Numeric conversion
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print("Feature shape:", X.shape)
print("Training normal-only Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X)

joblib.dump(model, output_path)

print("Model saved at:", output_path)
print("NORMAL-ONLY ISOLATION FOREST TRAINING COMPLETED")