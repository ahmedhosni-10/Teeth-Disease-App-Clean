import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---

# 1. Hugging Face Configuration (The new part)
REPO_ID = "ahmedhosni10/teeth-disease-model"
FILENAME = "teeth_best_model_vgg16.h5"

# 2. Your Model's Configuration (Kept from your code)
IMG_SIZE = (224, 224)
CLASS_NAMES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]


# --- MODEL LOADING (Updated to use Hugging Face) ---

# Use Streamlit's caching to download and load the model only once.
@st.cache_resource
def load_model_from_hf(repo_id, filename):
    """
    Downloads the model from Hugging Face Hub, caches it, and returns the loaded model.
    """
    with st.spinner("Downloading and loading model... this may take a moment."):
        try:
            # Download the model from the Hub
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            # Load the model using Keras
            model = tf.keras.models.load_model(model_path)
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model from Hugging Face: {e}")
            return None

# Load the model by calling the function
model = load_model_from_hf(REPO_ID, FILENAME)


# --- HELPER FUNCTIONS (Kept from your code) ---

def preprocess_image(image):
    """Preprocesses the uploaded image to match the model's input requirements."""
    # Resize the image
    image = image.resize(IMG_SIZE)
    # Convert image to numpy array
    image_array = np.array(image)
    # Ensure image is in RGB format (for PNGs with alpha channels)
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    # Expand dimensions to create a batch of 1
    image_array = np.expand_dims(image_array, axis=0)
    # Normalize the image data
    image_array = image_array / 255.0
    return image_array


# --- STREAMLIT APP LAYOUT (Kept from your code) ---

st.title("🦷 Teeth Disease Classification")
st.write("Upload an image of a tooth, and the model will predict its condition.")
st.write("This app uses a VGG16-based deep learning model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Create a button to trigger the classification
    if st.button('Classify Image'):
        with st.spinner('Analyzing the image...'):
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Make a prediction
            prediction = model.predict(processed_image)
            
            # Get the predicted class index and confidence
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Get the class name from your list
            predicted_class_name = CLASS_NAMES[predicted_index]

            st.success(f"**Prediction:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence:.2f}%")

elif model is None:
    st.warning("Model could not be loaded. Please check the logs for errors.")

st.sidebar.header("About")
st.sidebar.info("This is a web application for classifying teeth diseases. It was built using a fine-tuned VGG16 model and deployed with Streamlit.")
