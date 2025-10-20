import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

st.title("🖊️ MNIST Digit Classifier")
st.write("Upload a handwritten digit image (28x28 grayscale preferred).")

uploaded_file = st.file_uploader("Choose image...", type=["png","jpg","jpeg"])
model = load_model("mnist_cnn.h5")  # ensure model file exists

def preprocess(img):
    img = img.convert('L').resize((28,28))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, (0,-1))
    return arr

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Input Image', use_column_width=True)
    x = preprocess(img)
    pred = model.predict(x).argmax(axis=1)[0]
    st.success(f"Predicted Digit: **{pred}**")

