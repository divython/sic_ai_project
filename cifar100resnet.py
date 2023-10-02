# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:19:51 2023

@author: Divyanshu
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions, EfficientNetB0
import requests

# Load pre-trained EfficientNetB0 model
model = EfficientNetB0(weights='imagenet')

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
imagenet_labels = requests.get(LABELS_URL).json()

# Load and preprocess the input image
def preprocess_image(image_path):
    img = tf_image.load_img(image_path, target_size=(224, 224))
    x = tf_image.img_to_array(img)
    x = preprocess_input(x)
    x = tf.expand_dims(x, axis=0)
    return x

# Perform inference
def predict(image):
    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0]

# Streamlit app
st.title("EfficientNetB0 Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Display ImageNet labels in a box with a scrollbar
st.sidebar.header("ImageNet Class Labels")
rows_to_display = st.sidebar.slider("Number of Rows to Display", 1, 20, 10)
displayed_labels = imagenet_labels[:rows_to_display]
st.sidebar.text_area("Labels", "\n".join(displayed_labels), height=200)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Display the uploaded image
    image = preprocess_image(uploaded_file)
    st.image(np.array(image[0], dtype=np.uint8), caption="Classifying Image.", use_column_width=True)

    # Perform inference and display the result
    predictions = predict(image)
    for pred in predictions:
        st.write(f"Prediction: {pred[1]} (Confidence: {pred[2]:.4f})")
