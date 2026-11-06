import tensorflow as tf
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "val"
TEST_DIR = BASE_DIR / "data" / "test"

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# 1. Load datasets

def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names

    assert class_names == val_ds.class_names
    assert class_names == test_ds.class_names

    return train_ds, val_ds, test_ds, class_names


# 2. Data augmentation

augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.1),

    tf.keras.layers.RandomZoom(0.1),

    tf.keras.layers.RandomContrast(0.1)

])


# 3. MobileNetV2 preprocessing

def preprocess_train(images, labels):

    images = augmentation(images, training=True)

    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        images
    )

    return images, labels


def preprocess_val_test(images, labels):

    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        images
    )

    return images, labels


# 4. Complete pipeline

def prepare_datasets():

    train_ds, val_ds, test_ds, class_names = load_datasets()

    train_ds = train_ds.map(
        preprocess_train,
        num_parallel_calls=AUTOTUNE
    )

    val_ds = val_ds.map(
        preprocess_val_test,
        num_parallel_calls=AUTOTUNE
    )

    test_ds = test_ds.map(
        preprocess_val_test,
        num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# 5. Test preprocessing

if __name__ == "__main__":

    train_ds, val_ds, test_ds, class_names = prepare_datasets()

    print("\nClasses:")
    print(class_names)

    print("\nNumber of classes:", len(class_names))

    for images, labels in train_ds.take(1):

        print("\nImage batch shape:", images.shape)

        print("Label batch shape:", labels.shape)

        print("Minimum pixel value:", tf.reduce_min(images).numpy())

        print("Maximum pixel value:", tf.reduce_max(images).numpy())

    print("\nPreprocessing completed successfully!")