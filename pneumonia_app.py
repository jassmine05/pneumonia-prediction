import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import base64

# Function to encode image to base64
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Paths to your images (Ensure correct paths)
main_bg_path = "mainimg1.jpg"  
sidebar_bg_path = "sb2.jpg"  

# Encode images to base64
main_bg = get_base64_of_bin_file(main_bg_path)
sidebar_bg = get_base64_of_bin_file(sidebar_bg_path)

# Inject CSS for background styling
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{main_bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    background-image: url("data:image/jpeg;base64,{sidebar_bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-blend-mode: multiply;
    background-color: rgba(255, 255, 255, 0.8);
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and description
st.title('Pneumonia Detection')
st.subheader("This app predicts *PNEUMONIA DISEASE* using Deep Learning")

# Load the trained model
try:
    model = load_model('vgg19_model.h5')
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Sidebar for image upload
st.sidebar.header("Upload Chest X-ray Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)

    # Resize image to 224x224 (model input size)
    image_resized = cv2.resize(image, (224, 224))

    # Convert grayscale to RGB (if needed)
    if len(image_resized.shape) == 2:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)

    # Normalize image (Scale pixel values between 0 and 1)
    image_resized = image_resized / 255.0

    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    image_resized = np.expand_dims(image_resized, axis=0)

    return image_resized

# If an image is uploaded
if uploaded_file:
    st.success("Image uploaded successfully. Ready for prediction!")

    # Load and display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image (Processed)", use_column_width=True)

    # Preprocess image
    img_preprocessed = preprocess_image(img)

    # Button to trigger prediction
    if st.button("Predict"):
        try:
            # Predict class
            prediction = model.predict(img_preprocessed)
            prediction_class = np.argmax(prediction)

            # Pneumonia classification labels
            labels = np.array(['Normal', 'Pneumonia'])

            # Display result
            st.subheader('Prediction')
            st.write(f'Predicted Class: *{labels[prediction_class]}*')

            st.subheader('Prediction Probabilities')
            st.write(prediction)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Waiting for image upload...")