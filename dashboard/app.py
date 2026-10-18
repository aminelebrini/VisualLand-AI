from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf


CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]
MODEL_PATH = ROOT_DIR / "models" / "intel_image_classifier_model.keras"
HISTORY_PATH = ROOT_DIR / "results" / "history_data_model.json"
TRAIN_DIR = ROOT_DIR / "data" / "Intel_Image_Classification" / "seg_train_test" / "seg_train"


st.set_page_config(
    page_title="VisualLand AI",
    page_icon="🌍",
    layout="wide",
)


def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_dataset_counts() -> pd.DataFrame:
    counts = {}
    if TRAIN_DIR.exists():
        for class_name in CLASS_NAMES:
            class_dir = TRAIN_DIR / class_name
            counts[class_name] = (
                sum(1 for path in class_dir.iterdir() if path.is_file())
                if class_dir.exists()
                else 0
            )
    else:
        counts = {class_name: 0 for class_name in CLASS_NAMES}
    return pd.DataFrame({"Classe": list(counts), "Images": list(counts.values())})


@st.cache_data
def get_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    with HISTORY_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def predict_image(model, image: Image.Image) -> tuple[str, float, pd.DataFrame]:
    image_array = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
    image_array /= 255.0
    probabilities = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    ranking = np.argsort(probabilities)[::-1]
    probability_table = pd.DataFrame(
        {
            "Classe": [CLASS_NAMES[index] for index in ranking],
            "Confiance": [float(probabilities[index]) for index in ranking],
        }
    )
    return (
        CLASS_NAMES[int(ranking[0])],
        float(probabilities[ranking[0]]),
        probability_table,
    )


def show_overview():
    counts = get_dataset_counts()
    history = get_history()
    model_available = MODEL_PATH.exists()

    st.title("🌍 VisualLand AI")
    st.subheader("Classification d'images de paysages")
    st.caption("Dashboard de suivi du dataset et de prédiction avec le modèle CNN.")

    total_images = int(counts["Images"].sum())
    epochs = len(history.get("loss", []))
    metric_columns = st.columns(4)
    metric_columns[0].metric("Images", f"{total_images:,}")
    metric_columns[1].metric("Classes", len(CLASS_NAMES))
    metric_columns[2].metric("Époques", epochs or "N/A")
    metric_columns[3].metric("Modèle", "Disponible" if model_available else "Absent")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Distribution du dataset")
        st.bar_chart(counts.set_index("Classe")["Images"])
        st.dataframe(counts, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Courbes d'entraînement")
        if history:
            history_frame = pd.DataFrame(history)
            available_columns = [
                column
                for column in ["loss", "val_loss", "accuracy", "val_accuracy"]
                if column in history_frame
            ]
            st.line_chart(history_frame[available_columns])
        else:
            st.info("Aucun historique trouvé. Lancez d'abord `main.py`.")


def show_prediction():
    st.header("Prédire une image")
    model = get_model()
    if model is None:
        st.error(
            "Modèle introuvable. Lancez `.venv/bin/python main.py` "
            "pour entraîner et sauvegarder le modèle."
        )
        return

    uploaded_file = st.file_uploader(
        "Choisissez une image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is None:
        st.info("Importez une image pour obtenir une prédiction.")
        return

    image = Image.open(uploaded_file)
    preview, result = st.columns(2)
    with preview:
        st.image(image, caption="Image importée", use_container_width=True)
    with result:
        predicted_class, confidence, probabilities = predict_image(model, image)
        st.success(f"Classe prédite : {predicted_class}")
        st.metric("Confiance", f"{confidence:.2%}")
        st.dataframe(
            probabilities.style.format({"Confiance": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )


def main():
    with st.sidebar:
        st.title("VisualLand AI")
        page = st.radio("Navigation", ["Vue d'ensemble", "Prédiction"])
        st.caption("CNN · 6 classes · images redimensionnées en 256×256")

    if page == "Vue d'ensemble":
        show_overview()
    else:
        show_prediction()


if __name__ == "__main__":
    main()
