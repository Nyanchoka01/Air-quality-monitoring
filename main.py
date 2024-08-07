import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D
from PIL import Image
import os
import pandas as pd
import requests
# Set the title of the Streamlit web application
st.markdown('<h1 style="color: blue;">AIR QUALITY MONITORING BY UNIVERSITY OF NAIROBI,DEPARTMENT OF ELECTRICAL AND INFORMATION ENGINEERING</h1>', unsafe_allow_html=True)

# Define the model structure (example with one Conv2D layer)

# Load the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))
    # Add more layers as needed
])
def delete_image(image_file):
    image_path = os.path.join(UPLOAD_DIR, image_file)
    os.remove(image_path)
    st.write(f"Deleted:")
# Path to the saved model
model_path = 'https://drive.google.com/file/d/1GBY0y4kri03e4ScTjBaZ6S-uYBzZMcZa/view?usp=drive_link'

def download_model():
    url = model_path  # URL to your model file in the release
    local_filename = "model_file"  # Local path to save the model file

    if not os.path.exists(local_filename):
        with st.spinner('Downloading model...'):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f:
                for data in response.iter_content(1024):
                    f.write(data)

    return local_filename

model_path = download_model()
# Load your model here using the model_path
model = load_model(model_path)

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, resize=True):
    img = Image.open(img_path)
    if resize:
        img = img.resize((224, 224))  # Resize the image to 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array, img

# Class labels
class_labels = ["Good", "Moderate", "Unhealthy_for_Sensitive_Groups", "Unhealthy", "Very_Unhealthy", "Severe"]

# Load the model
try:
    model = load_model(model_path)
    st.write("Model loaded successfully!")
except ValueError as e:
    st.write(f"Error loading model: {e}")
    exit()

# Streamlit file uploader
uploaded_files = st.file_uploader("Choose images to predict the air quality...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
UPLOAD_DIR = "/"

if uploaded_files:
    st.write(f"Number of images uploaded: {len(uploaded_files)}")
    # Calculate how many rows are needed
    num_rows = (len(uploaded_files) + 2) // 3  # +2 to ensure rounding up
    image_index = 0
    j = 0
    # Create rows of images
    for _ in range(num_rows):
        cols = st.columns(3)  # Creates a row of 3 columns
        for col in cols:
            if image_index < len(uploaded_files):
                uploaded_file = uploaded_files[image_index]
                img_array, img = load_and_preprocess_image(uploaded_file)

                # Display the image in a column
                col.image(img, caption='Uploaded Image')

                # Make predictions

                predictions = model.predict(img_array)[0]
                predicted_class = class_labels[np.argmax(predictions)]
                # df = pd.read_csv("Datasets/data.csv")
                # df = df[df['AQI'] == predicted_class]
                # output_string=("\n".join([f"{col}: {value}" for col, value in df.iloc[0].items()]))
                # st.markdown(f"```\n{output_string}\n```")

                accuracy = np.max(predictions)

                file_path = os.path.join(UPLOAD_DIR, predicted_class+"@"+f'{accuracy*100:.2f}%'+"@"+uploaded_file.name)
                if not os.path.exists(file_path):
                    # Save the uploaded file if it doesn't already exist
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.write(f"Saved: {uploaded_file.name}")
                else:
                    st.write(f"File already exists and was not saved: {uploaded_file.name}")

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Display predictions in the same column
                col.write(f"Predicted Class: {predicted_class}")
                col.write(f"Accuracy: {accuracy* 100:.2f}%")
                image_index += 1

image_files = os.listdir(UPLOAD_DIR)
st.title("Sample Images")
if image_files:
    # Define the number of columns
    num_cols = 3

    # Create a grid layout
    cols = st.columns(num_cols)
    # Resize the image

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(UPLOAD_DIR, image_file)
        image = Image.open(image_path)
        # Display each image in a grid
        size = min(image.size)
        image = image.resize((size, size))
        with cols[i % num_cols]:
            st.image(image, caption=" ".join(image_file.split('@')[0].split("_"))+ "    \nAccuracy   "+image_file.split('@')[1], use_column_width=True)
            if st.button(f"Delete",key = image_file):
                delete_image(image_file)
            try:
                df = pd.read_csv("data.csv")
                df = df[df['AQI'] == image_file.split('@')[0]]
                output_string = ("\n".join([f"{col}: {value}" for col, value in df.iloc[0].items() if col !="AQI"]))
                st.markdown(f"```\n{output_string}\n```")
            except:
                st.write("Unrecognised class")

else:

    st.write("No images uploaded yet.")

