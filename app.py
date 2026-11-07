"""Streamlit portfolio demo for medicinal plant image classification."""

from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import load_predictor, predict_image

st.set_page_config(page_title="Medicinal Plant Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Medicinal Plant Classifier")
st.write("Upload a plant image to see the model's top predictions.")

checkpoint_path = Path(st.sidebar.text_input("Checkpoint path", "artifacts/medicinal_plant_hybrid.pt"))
uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if not checkpoint_path.is_file():
        st.info("Train the model first; the checkpoint will appear at the path shown in the sidebar.")
    elif st.button("Classify plant", type="primary"):
        with st.spinner("Loading model and classifying image..."):
            @st.cache_resource
            def get_predictor(path: str):
                return load_predictor(path)

            model, transform, class_names, device = get_predictor(str(checkpoint_path))
            predictions = predict_image(image, model, transform, class_names, device)
        st.subheader(f"Best match: {predictions[0][0]}")
        st.progress(predictions[0][1], text=f"Confidence: {predictions[0][1]:.1%}")
        st.write("Top predictions")
        for name, confidence in predictions:
            st.write(f"{name}: {confidence:.1%}")
        st.caption("Predictions are experimental and are not medical advice or a substitute for expert identification.")
else:
    st.caption("Checkpoint: " + str(checkpoint_path))
