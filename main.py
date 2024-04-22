import tensorflow as tf
import streamlit as st
import numpy as np
from operator import itemgetter
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model , Sequential
from tensorflow.keras.layers import Conv2D
from collections import Counter
# Load the model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))
    # Add more layers as needed
])
model_path = 'C:\\Users\\Dione Nyanchoka\\Downloads\\augm.h5'
model = load_model(model_path)

# Load and preprocess the test image
img_path ='C:\\Users\\Dione Nyanchoka\\Desktop\\All_img\\BIR_MOD_2023-02-13-09.30-1-14.jpg'


def load_and_preprocess_image(img_path):
  img = image.load_img(img_path, target_size=(224, 224))  # Example target size, adjust according to your model
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch dimension
  img_array /= 255.  # Normalize pixel values
  return img_array
# Class labels
class_labels = ["Good","Moderate","Unhealthy_for_Sensitive_Groups","Unhealthy","Very_Unhealthy","Severe"]

try:
  # Load the model
  model = load_model(model_path)
  print("Model loaded successfully!")
except ValueError as e:
  # Handle error if model cannot be loaded (e.g., file not found)
  print(f"Error loading model: {e}")
  exit()

# Load and preprocess the image
uploaded_files = st.file_uploader("Choose images to predict the air quality...", type=['jpg', 'jpeg', 'png'],
                        accept_multiple_files=True)
save_dir = "images"
# Process each uploaded file
for uploaded_file in uploaded_files:
    # To read file as bytes:
    img_array = load_and_preprocess_image(uploaded_file)

    # Make predictions
    #st.write(([(i,prob) for i,prob in enumerate(model.predict(img_array)[0])]))
    predictions = {clas: str(int(prob * 100)) + "%" for clas, prob in zip(class_labels, model.predict(img_array)[0])}
    #st.write(Counter(predictions))
    preds = [str("%.4f" % prob[1])+"%"+str(prob[0])+"%"+clas for prob,clas in zip([(i,prob) for i,prob in enumerate(model.predict(img_array)[0])],class_labels)]

    key_with_highest_value = max([i[:3] for i in preds])
    max_predictions = [pred for pred in preds if key_with_highest_value == pred[:3]]
    st.write(max_predictions[0].split("%")[2], str(float(max_predictions[0].split("%")[0])*100) + " %        ",[i for i in uploaded_file.name.split("_")][1])


