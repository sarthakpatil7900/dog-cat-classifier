import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cat_dog_cnn_model.h5")

# Title
st.title("🐶 Dog vs 🐱 Cat Classifier")
st.write("Upload an image of a cat or dog, and I'll tell you which one it is!")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((32, 32))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show result
    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {confidence:.2f}")
