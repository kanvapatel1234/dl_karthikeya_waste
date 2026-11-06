import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# PATHS AND CONFIGURATION
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
TEST_DIR = BASE_DIR / "data" / "test"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# -------------------------------
# LOAD MODEL
# -------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# -------------------------------
# LOAD TEST DATASET
# -------------------------------

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Number of classes:", num_classes)


# -------------------------------
# PREPROCESS TEST IMAGES
# -------------------------------

def preprocess(images, labels):

    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)

    return images, labels


test_ds = test_ds.map(preprocess)


# -------------------------------
# MODEL PREDICTIONS
# -------------------------------

print("\nGenerating predictions...")

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)


# -------------------------------
# CALCULATE METRICS
# -------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision_macro = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)

recall_macro = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

f1_macro = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

precision_weighted = precision_score(
    y_true, y_pred, average="weighted", zero_division=0
)

recall_weighted = recall_score(
    y_true, y_pred, average="weighted", zero_division=0
)

f1_weighted = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)


# -------------------------------
# CLASSIFICATION REPORT
# -------------------------------

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_text = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    zero_division=0
)


# -------------------------------
# PRINT RESULTS
# -------------------------------

print("\n" + "=" * 55)
print("       WASTE CLASSIFICATION MODEL EVALUATION")
print("=" * 55)

print(f"\nTotal Test Images : {len(y_true)}")
print(f"Test Accuracy    : {accuracy * 100:.2f}%")

print("\nMACRO AVERAGE")
print(f"Precision        : {precision_macro:.4f}")
print(f"Recall           : {recall_macro:.4f}")
print(f"F1 Score         : {f1_macro:.4f}")

print("\nWEIGHTED AVERAGE")
print(f"Precision        : {precision_weighted:.4f}")
print(f"Recall           : {recall_weighted:.4f}")
print(f"F1 Score         : {f1_weighted:.4f}")

print("\nCLASSIFICATION REPORT")
print(report_text)


# -------------------------------
# SAVE METRICS AS JSON
# -------------------------------

metrics = {
    "test_images": int(len(y_true)),
    "number_of_classes": int(num_classes),
    "accuracy": float(accuracy),
    "macro_precision": float(precision_macro),
    "macro_recall": float(recall_macro),
    "macro_f1": float(f1_macro),
    "weighted_precision": float(precision_weighted),
    "weighted_recall": float(recall_weighted),
    "weighted_f1": float(f1_weighted),
    "class_names": class_names,
    "classification_report": report
}

with open(RESULTS_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

with open(RESULTS_DIR / "classification_report.txt", "w") as file:
    file.write(report_text)


# -------------------------------
# CONFUSION MATRIX
# -------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes)
)

plt.figure(figsize=(13, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Waste Classification - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# -------------------------------
# CLASS-WISE PERFORMANCE GRAPH
# -------------------------------

class_precision = [
    report[name]["precision"] for name in class_names
]

class_recall = [
    report[name]["recall"] for name in class_names
]

class_f1 = [
    report[name]["f1-score"] for name in class_names
]

x = np.arange(num_classes)
width = 0.25

plt.figure(figsize=(14, 7))

plt.bar(x - width, class_precision, width, label="Precision")
plt.bar(x, class_recall, width, label="Recall")
plt.bar(x + width, class_f1, width, label="F1-score")

plt.xticks(x, class_names, rotation=45, ha="right")
plt.ylim(0, 1.1)

plt.title("Class-wise Model Performance")
plt.xlabel("Waste Categories")
plt.ylabel("Score")

plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_performance.png",
    dpi=300
)

plt.close()


print("\nEvaluation completed successfully!")

print("\nGenerated files:")
print("1. results/metrics.json")
print("2. results/classification_report.txt")
print("3. results/confusion_matrix.png")
print("4. results/class_performance.png")