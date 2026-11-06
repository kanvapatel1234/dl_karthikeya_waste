import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


IMG_SIZE = (224, 224)
NUM_CLASSES = 12


def build_model():

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Add custom classification head
    model = models.Sequential([

        layers.Input(shape=(224, 224, 3)),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.3),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ])

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()

    print("\nModel created successfully!")