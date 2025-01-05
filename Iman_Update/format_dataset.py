import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

train_json_path = "train_data.json"
unique_labels_path = "unique_labels.json"
images_folder = "/train_images_folder/re_512_3ch/Train"
output_folder = "Train_dataset"

os.makedirs(output_folder, exist_ok=True)

with open(train_json_path, "r") as f:
    train_data = json.load(f)

with open(unique_labels_path, "r") as f:
    unique_labels = json.load(f)

label_to_index = {label: i for i, label in enumerate(unique_labels)}

dataset = []
for entry in tqdm(train_data, desc="Processing JSON entries"):
    image_id = entry["id"]
    labels = entry["label"].strip("'").split("', '") if entry["label"] else []

    if not labels:
        continue

    label_vector = np.zeros(len(unique_labels), dtype=np.float32)
    for label in labels:
        if label in label_to_index:
          label_vector[label_to_index[label]] = 1

    image_path = os.path.join(images_folder, f"{image_id}.jpg")
    if os.path.exists(image_path):
        with Image.open(image_path) as img:

            img_gray = img.convert("L")

            output_image_path = os.path.join(output_folder, f"{image_id}.jpg")
            img_gray.save(output_image_path)

            dataset.append({"image_path": output_image_path, "label_vector": label_vector.tolist(), "counter": 0})

dataset_json_path = os.path.join(output_folder, "dataset.json")
with open(dataset_json_path, "w") as f:
    json.dump(dataset, f)

print(f"Processed dataset saved at {output_folder}")
