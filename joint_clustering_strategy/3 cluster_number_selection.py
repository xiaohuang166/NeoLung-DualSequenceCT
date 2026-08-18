import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# ---------- Training voxel features ----------
data_dir = r"F:\excel_training_voxel_features"

patient_data = []

for file in os.listdir(data_dir):

    if not file.lower().endswith((".xlsx", ".xls")):
        continue

    df = pd.read_excel(os.path.join(data_dir, file))

    # Clustering features
    features = df.iloc[:, 5:].astype(np.float32)
    patient_data.append(features.values)

# ---------- Combine training voxels ----------
train_data = np.concatenate(patient_data, axis=0)

# Remove voxels containing NaN
train_data = train_data[~np.isnan(train_data).any(axis=1)]

print("Training voxel data:", train_data.shape)

# ---------- Evaluate cluster numbers ----------
k_range = range(2, 11)

ch_scores = []
db_scores = []

for k in k_range:

    labels = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10
    ).fit_predict(train_data)

    ch_scores.append(
        calinski_harabasz_score(train_data, labels)
    )

    db_scores.append(
        davies_bouldin_score(train_data, labels)
    )

# Higher CH and lower DB indicate better clustering
best_ch_k = list(k_range)[np.argmax(ch_scores)]
best_db_k = list(k_range)[np.argmin(db_scores)]

print("Best K by CH:", best_ch_k)
print("Best K by DB:", best_db_k)

# ---------- CH curve ----------
plt.figure(figsize=(6, 4))
plt.plot(k_range, ch_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Index")
plt.title("Calinski-Harabasz Score")
plt.tight_layout()
plt.show()

# ---------- DB curve ----------
plt.figure(figsize=(6, 4))
plt.plot(k_range, db_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.title("Davies-Bouldin Score")
plt.tight_layout()
plt.show()
