import os
import re
import six
import SimpleITK as sitk
from radiomics import featureextractor

# ---------- Paths ----------
root = r"E:\RESULT\szl"
image_dir = os.path.join(root, "image_1mm")
mask_dir = os.path.join(root, "contrast_mask_1mm")
output_dir = os.path.join(root, "feature_map_7")
params_file = r"D:\CODE\voxel_radiomics.yaml"

os.makedirs(output_dir, exist_ok=True)

# ---------- PyRadiomics extractor ----------
extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
extractor.addProvenance(False)

# ---------- Voxel-wise feature extraction ----------
image_files = [
    f for f in os.listdir(image_dir)
    if re.match(r"^\d+b\.nrrd$", f, re.IGNORECASE)
]

for image_file in image_files:

    case_id = os.path.splitext(image_file)[0]
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, f"{case_id}.nrrd")

    if not os.path.exists(mask_path):
        print(f"Mask missing: {case_id}")
        continue

    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

        features = extractor.execute(image, mask, voxelBased=True)

        for name, value in six.iteritems(features):

            if not isinstance(value, sitk.Image):
                continue

            feature_dir = os.path.join(output_dir, name)
            os.makedirs(feature_dir, exist_ok=True)

            output_file = os.path.join(
                feature_dir,
                f"{name}_{case_id}.nrrd"
            )

            # Skip existing feature map
            if os.path.exists(output_file):
                continue

            sitk.WriteImage(value, output_file, True)

        print(f"Completed: {case_id}")

    except Exception as e:
        print(f"Error in {case_id}: {e}")
