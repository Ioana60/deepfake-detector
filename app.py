import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the model
model = load_model("deepfake_model.h5")

st.title("🧠 DeepFake Image Detector")
st.write("Upload an image of a face, and I will tell you if it's **Real** or **Fake**!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    if img_array.shape[-1] == 4:  # Remove alpha channel if exists
        img_array = img_array[:, :, :3]

    img_resized = cv2.resize(img_array, (224, 224)) / 255.0
    img_input = img_resized.reshape(1, 224, 224, 3)

    prediction = model.predict(img_input)[0][0]
    label = "🔴 Fake" if prediction > 0.5 else "🟢 Real"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: **{label}**")
    st.write(f"Confidence: `{confidence:.2%}`")
