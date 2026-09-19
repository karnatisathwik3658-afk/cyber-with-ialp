
import pandas as pd

dataset_path = r".\DataSet\kddcup.data.cleaned.txt"

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

print("Reading dataset in chunks...")

label_counts = {}

for chunk in pd.read_csv(
    dataset_path,
    names=columns,
    chunksize=100000
):
    counts = chunk["attack_type"].value_counts()

    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + count

print("\n========== ATTACK DISTRIBUTION ==========")

sorted_counts = sorted(
    label_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

for label, count in sorted_counts:
    print(f"{label}: {count}")

print("\nLABEL DISTRIBUTION CHECK COMPLETED")