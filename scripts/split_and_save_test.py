import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from PIL import Image

DATA_DIR = "data/raw/images"
IMG_SIZE = (48, 48)
BATCH_SIZE = 1  # for per-sample iteration

# Load entire validation dataset without shuffling, batch size 1
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_ds.class_names
print("Classes:", class_names)

# Collect all images and labels in lists
all_images = []
all_labels = []

for images, labels in val_ds:
    img = images[0].numpy().astype("uint8")
    label_idx = tf.argmax(labels[0]).numpy() if labels.shape[-1] > 1 else labels[0].numpy()
    all_images.append(img)
    all_labels.append(label_idx)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Create directory for test set
SAVE_DIR = Path(DATA_DIR) / "test"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Stratified split: select 10% from each class
test_indices = []

np.random.seed(42)
for cls_idx, cls_name in enumerate(class_names):
    cls_indices = np.where(all_labels == cls_idx)[0]
    n_test = int(len(cls_indices) * 0.1)
    selected = np.random.choice(cls_indices, size=n_test, replace=False)
    test_indices.extend(selected)

# Save test images by class
count = 0
for idx in test_indices:
    img = all_images[idx]
    cls_idx = all_labels[idx]
    cls_name = class_names[cls_idx]

    class_dir = SAVE_DIR / cls_name
    class_dir.mkdir(parents=True, exist_ok=True)

    img_pil = Image.fromarray(img)
    img_pil.save(class_dir / f"{count:06}.jpg")
    count += 1

print(f"✅ Saved {count} stratified test images to {SAVE_DIR}")
