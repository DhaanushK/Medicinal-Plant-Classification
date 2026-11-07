"""Streamlit portfolio demo for medicinal plant image classification."""

from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image

from src.predict import load_predictor, predict_image

MODEL_REPO_ID = "DhaanushK/medicinal-plant-classifier"
MODEL_FILENAME = "medicinal_plant_hybrid.pt"
LOCAL_CHECKPOINT = Path("artifacts/medicinal_plant_hybrid.pt")

st.set_page_config(page_title="Medicinal Plant Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Medicinal Plant Classifier")
st.write("Upload a plant image to see the model's top predictions.")


@st.cache_resource(show_spinner=False)
def get_predictor():
    # Use the local file during development. On Streamlit Cloud,
    # download the private checkpoint using the HF_TOKEN secret.
    if LOCAL_CHECKPOINT.is_file():
        checkpoint_path = LOCAL_CHECKPOINT
    else:
        token = st.secrets.get("HF_TOKEN")
        if not token:
            raise RuntimeError("The HF_TOKEN secret is not configured.")

        checkpoint_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            token=token,
        )

    return load_predictor(checkpoint_path)


uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify plant", type="primary"):
        try:
            with st.spinner("Loading the model and classifying the image..."):
                model, transform, class_names, device = get_predictor()
                predictions = predict_image(image, model, transform, class_names, device)
        except Exception as exc:
            st.error(
                "The classifier could not load its model. Check that the HF_TOKEN "
                "secret is saved in Streamlit and has read access to the model repository."
            )
            st.caption(f"Error type: {type(exc).__name__}")
        else:
            st.subheader(f"Best match: {predictions[0][0]}")
            st.progress(predictions[0][1], text=f"Confidence: {predictions[0][1]:.1%}")
            st.write("Top predictions")

            for name, confidence in predictions:
                st.write(f"{name}: {confidence:.1%}")

            st.caption(
                "Predictions are experimental and are not medical advice or a "
                "substitute for expert identification."
            )
else:
    st.caption("The trained classifier loads when you request a prediction.")
