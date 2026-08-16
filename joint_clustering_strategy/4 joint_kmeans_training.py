import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---------- Paths ----------
train_dir = r"F:\0624lung_data\szl\cluster\excel_training_voxel_features"
model_dir = r"F:\0624lung_data\szl\cluster\models"

os.makedirs(model_dir, exist_ok=True)

# ---------- Load training voxel features ----------
data = []

for file in os.listdir(train_dir):

    if not file.lower().endswith((".xlsx", ".xls")):
        continue

    df = pd.read_excel(os.path.join(train_dir, file))

    # Joint clustering: nCE HU, nCE JointEntropy, CE HU, CE JointEntropy
    features = df.iloc[:, 5:9].astype(np.float32)

    # Remove voxels containing NaN
    features = features.dropna()

    data.append(features.values)

train_data = np.concatenate(data, axis=0)

print("Training data shape:", train_data.shape)

# ---------- Train K-means ----------
kmeans = KMeans(
    n_clusters=3,
    random_state=0,
    n_init=10
)

kmeans.fit(train_data)

# ---------- Save model ----------
model_path = os.path.join(model_dir, "joint_kmeans_k3.pkl")
joblib.dump(kmeans, model_path)

print("Model saved:", model_path)
