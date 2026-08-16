import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------- Paths ----------
mask_dir = r"F:\0624lung_data\gd\msk"
feature_root = r"F:\0624lung_data\gd\large_feature_map7_2"
output_dir = r"F:\0624lung_data\gd\cluster\excel_singlefeichuang"

os.makedirs(output_dir, exist_ok=True)


def get_patient_id(filename):
    match = re.match(r"^(\d+)", filename)
    return match.group(1) if match else None


def align_feature_map(feature_img, mask_img):
    """Align the cropped feature map to the original mask space."""
    return sitk.Resample(
        feature_img,
        mask_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        np.nan,
        sitk.sitkFloat32
    )


# ---------- Process each patient ----------
for mask_file in os.listdir(mask_dir):

    if not mask_file.lower().endswith(".nrrd"):
        continue

    patient_id = get_patient_id(mask_file)
    if patient_id is None:
        continue

    output_file = os.path.join(output_dir, f"{patient_id}.xlsx")

    if os.path.exists(output_file):
        continue

    mask_img = sitk.ReadImage(os.path.join(mask_dir, mask_file))
    mask = sitk.GetArrayFromImage(mask_img) > 0

    z, y, x = np.where(mask)

    data = pd.DataFrame({
        "z": z,
        "y": y,
        "x": x
    })

    # ---------- Load voxel-wise feature maps ----------
    for feature_name in os.listdir(feature_root):

        feature_dir = os.path.join(feature_root, feature_name)

        if not os.path.isdir(feature_dir):
            continue

        pattern = rf"^{re.escape(feature_name)}_{patient_id}[bc].*\.nrrd$"

        matches = [
            f for f in os.listdir(feature_dir)
            if re.match(pattern, f, re.IGNORECASE)
        ]

        if not matches:
            continue

        feature_img = sitk.ReadImage(
            os.path.join(feature_dir, matches[0])
        )

        feature_img = align_feature_map(feature_img, mask_img)
        feature_array = sitk.GetArrayFromImage(feature_img)

        data[feature_name] = feature_array[mask]

    # ---------- Save ----------
    data.to_excel(output_file, index=False)

    print(f"Completed: {patient_id}")

print("Finished.")
