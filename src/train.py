import tensorflow as tf
from pathlib import Path

from preprocessing import prepare_datasets
from model import build_model


# Configuration
EPOCHS = 20
LEARNING_RATE = 0.001

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# Load datasets
train_ds, val_ds, test_ds, class_names = prepare_datasets()

print("\nClasses:", class_names)
print("Total classes:", len(class_names))


# Build model
model = build_model()


# Compile model
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# Callbacks
callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    )

]


# Train model
print("\nStarting training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks

)


# Save final model
model.save(MODEL_DIR / "final_model.keras")


print("\nTraining completed!")

print("Best model saved at:", MODEL_DIR / "best_model.keras")

print("Final model saved at:", MODEL_DIR / "final_model.keras")
