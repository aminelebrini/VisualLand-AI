import json
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from models.CNN import CNN
from src.DataTraitement import DataTraitement
from src.DatasetExplorer import DatasetExplorer


CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]


def create_labels(image_paths):
    return [CLASS_NAMES.index(image_path.parent.name) for image_path in image_paths]


def save_training_history(history):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    history_path = results_dir / "history_data_model.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file)
    print(f"Training history saved to {history_path}")


def prediction(model, test_dataset):
    true_labels = []
    predicted_labels = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        predicted_labels.extend(tf.argmax(predictions, axis=1).numpy())
        true_labels.extend(labels.numpy())

    recall = recall_score(true_labels, predicted_labels, average="weighted")
    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels)

    print(f"Test recall: {recall:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    print("Test confusion matrix:")
    print(matrix)


def main():
    dataset_root = Path("data/Intel_Image_Classification")
    train_dir = dataset_root / "seg_train_test" / "seg_train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    dataset_explorer = DatasetExplorer(train_dir)
    class_count, image_distribution = (
        dataset_explorer.identify_classes_and_distribution()
    )
    print("Class count:", class_count)
    print("Image distribution:", image_distribution)

    min_width, min_height = dataset_explorer.get_dimentions_format()
    print(f"Smallest image dimensions: {min_width}x{min_height}")

    duplicates, unique_images, _ = dataset_explorer.problematics_data()
    print(f"Duplicate images: {len(duplicates)}")
    print(f"Unique images: {len(unique_images)}")

    train, validation, test = dataset_explorer.separation_data(unique_images)
    print(
        f"Split sizes - train: {len(train)}, "
        f"validation: {len(validation)}, test: {len(test)}"
    )

    data_processing = DataTraitement()
    resized_train = data_processing.redimensionnement(train)
    resized_validation = data_processing.redimensionnement(validation)
    resized_test = data_processing.redimensionnement(test)

    normalized_train = data_processing.normalisation(resized_train)
    normalized_validation = data_processing.normalisation(resized_validation)
    normalized_test = data_processing.normalisation(resized_test)

    train_dataset = data_processing.chagement_par_batch(
        normalized_train, create_labels(train)
    )
    validation_dataset = data_processing.chagement_par_batch(
        normalized_validation, create_labels(validation)
    )
    test_dataset = data_processing.chagement_par_batch(
        normalized_test, create_labels(test)
    )
    augmented_train_dataset = data_processing.data_augmentation(train_dataset)

    cnn_model = CNN()
    model = cnn_model.model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    )

    history = cnn_model.training(
        augmented_train_dataset,
        validation_data=validation_dataset,
        callbacks=[early_stopping],
    )
    save_training_history(history)

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    prediction(model, test_dataset)

    model_path = Path("models/intel_image_classifier_model.keras")
    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
