# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 03:49:53 2023

@author: Divyanshu
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('mycifar10.keras')
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title('CIFAR-10 Image Classification')
st.write('Upload a picture for classification')
st.sidebar.title('Class Labels')
st.sidebar.write(labels)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict the class
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]

    st.write("Prediction: ", predicted_label)
