import os
import shutil
import random
from pathlib import Path

DATA_DIR = Path("data/raw/images")
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
TEST_SPLIT = 0.1  # 10%

random.seed(42)
TEST_DIR.mkdir(parents=True, exist_ok=True)

class_names = sorted([d.name for d in VALIDATION_DIR.iterdir() if d.is_dir()])
print("Classes:", class_names)

total_moved = 0

for class_name in class_names:
    val_class_dir = VALIDATION_DIR / class_name
    test_class_dir = TEST_DIR / class_name
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # List all images in validation class folder
    images = list(val_class_dir.glob("*"))
    n_test = int(len(images) * TEST_SPLIT)
    print(f"Class '{class_name}': moving {n_test} of {len(images)} images")

    # Randomly select images for test
    test_images = random.sample(images, n_test)

    for img_path in test_images:
        dest_path = test_class_dir / img_path.name
        # Move file from validation to test folder
        shutil.move(str(img_path), str(dest_path))
        total_moved += 1

print(f"✅ Moved total {total_moved} images from validation to test folder.")

# Ran this script and got 
# Classes: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Class 'angry': moving 96 of 960 images
# Class 'disgust': moving 11 of 111 images
# Class 'fear': moving 101 of 1018 images
# Class 'happy': moving 182 of 1825 images
# Class 'neutral': moving 121 of 1216 images
# Class 'sad': moving 113 of 1139 images
# Class 'surprise': moving 79 of 797 images
# ✅ Moved total 703 images from validation to test folder.