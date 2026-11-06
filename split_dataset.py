import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("data")

CLASSES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

for category in CLASSES:

    category_path = SOURCE_DIR / category

    images = [
        file for file in category_path.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # 70% training, 30% remaining
    train, remaining = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # Split remaining into validation and test
    val, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=42
    )

    splits = {
        "train": train,
        "val": val,
        "test": test
    }

    for split_name, split_images in splits.items():

        destination = OUTPUT_DIR / split_name / category
        destination.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, destination / image.name)

    print(
        f"{category}: "
        f"Train={len(train)}, "
        f"Val={len(val)}, "
        f"Test={len(test)}"
    )

print("\nDataset splitting completed!")