import os
import ast
import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------- Paths ----------
train_dir = r"E:\excel_training_voxel_features"
model_path = r"E:\joint_kmeans_k3.pkl"
output_dir = r"E:\training_habitat_maps"

os.makedirs(output_dir, exist_ok=True)

# ---------- Load training-derived K-means model ----------
kmeans = joblib.load(model_path)

n_clusters = kmeans.n_clusters
cluster_col = f"cluster{n_clusters}_label"


# ---------- Process each patient ----------
for file in os.listdir(train_dir):

    if not file.lower().endswith((".xlsx", ".xls")):
        continue

    excel_path = os.path.join(train_dir, file)
    nrrd_path = os.path.join(
        output_dir,
        os.path.splitext(file)[0] + ".nrrd"
    )

    df = pd.read_excel(excel_path)

    # Skip completed cases
    if cluster_col in df.columns and os.path.exists(nrrd_path):
        continue

    # First 5 columns:
    # Direction, Origin, Spacing, Dimension, Coordinate
    # Columns 6-9:
    # nCE HU, nCE JointEntropy, CE HU, CE JointEntropy
    features = df.iloc[:, 5:9].astype(np.float32)

    # Remove voxels containing NaN or Inf
    valid = np.isfinite(features.values).all(axis=1)

    if not valid.any():
        print(f"No valid voxels: {file}")
        continue

    # ---------- Cluster assignment ----------
    labels = kmeans.predict(features.values[valid])

    # Save 0-based cluster labels to Excel
    cluster_labels = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )
    cluster_labels.loc[valid] = labels

    df[cluster_col] = cluster_labels
    df.to_excel(excel_path, index=False)

    # ---------- Reconstruct 3D habitat map ----------
    direction = ast.literal_eval(str(df.iloc[0, 0]))
    origin = ast.literal_eval(str(df.iloc[0, 1]))
    spacing = ast.literal_eval(str(df.iloc[0, 2]))
    size = ast.literal_eval(str(df.iloc[0, 3]))

    # Coordinates are stored as (z, y, x)
    coordinates = np.array([
        ast.literal_eval(str(x))
        for x in df.loc[valid, df.columns[4]]
    ], dtype=int)

    # SimpleITK size = (x, y, z)
    habitat = np.zeros(
        (size[2], size[1], size[0]),
        dtype=np.uint8
    )

    # Background = 0; habitats = 1, 2, 3
    habitat[
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2]
    ] = labels + 1

    # ---------- Save NRRD ----------
    habitat_img = sitk.GetImageFromArray(habitat)
    habitat_img.SetSpacing(tuple(spacing))
    habitat_img.SetOrigin(tuple(origin))
    habitat_img.SetDirection(tuple(direction))

    sitk.WriteImage(habitat_img, nrrd_path)

    print(f"Completed: {file}")

print("All finished.")
