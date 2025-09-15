# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np

# # --- CONFIGURATION ---
# MODEL_PATH = 'teeth_best_model_vgg16.h5'
# IMG_SIZE = (224, 224)

# # IMPORTANT: Update this list with the class names your model was trained on.
# # The order must be the same as the one used during training.
# # You can find the order from your train_generator.class_indices
# CLASS_NAMES = ["CaS", "CoS", "Gum", "MC","OC","OLP","OT"] 


# # --- MODEL LOADING ---

# # Use Streamlit's caching to load the model only once.
# @st.cache_resource
# def load_model():
#     """Loads the trained Keras model."""
#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# model = load_model()


# # --- HELPER FUNCTIONS ---

# def preprocess_image(image):
#     """Preprocesses the uploaded image to match the model's input requirements."""
#     # Resize the image
#     image = image.resize(IMG_SIZE)
#     # Convert image to numpy array
#     image_array = np.array(image)
#     # Ensure image is in RGB format (for PNGs with alpha channels)
#     if image_array.shape[2] == 4:
#         image_array = image_array[:, :, :3]
#     # Expand dimensions to create a batch of 1
#     image_array = np.expand_dims(image_array, axis=0)
#     # Normalize the image data (if your model was trained on normalized data)
#     image_array = image_array / 255.0
#     return image_array


# # --- STREAMLIT APP LAYOUT ---

# st.title("🦷 Teeth Disease Classification")
# st.write("Upload an image of a tooth, and the model will predict its condition.")
# st.write("This app uses a VGG16-based deep learning model.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None and model is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")

#     # Create a button to trigger the classification
#     if st.button('Classify Image'):
#         with st.spinner('Analyzing the image...'):
#             # Preprocess the image
#             processed_image = preprocess_image(image)
            
#             # Make a prediction
#             prediction = model.predict(processed_image)
            
#             # Get the predicted class index and confidence
#             predicted_index = np.argmax(prediction)
#             confidence = np.max(prediction) * 100
            
#             # Get the class name from your list
#             predicted_class_name = CLASS_NAMES[predicted_index]

#             st.success(f"**Prediction:** {predicted_class_name}")
#             st.info(f"**Confidence:** {confidence:.2f}%")

# st.sidebar.header("About")
# st.sidebar.info("This is a web application for classifying teeth diseases. It was built using a fine-tuned VGG16 model and deployed with Streamlit.")
import streamlit as st
import tensorflow as tf
from urllib.request import urlretrieve # لاستيراد مكتبة التحميل
import os # للتعامل مع الملفات

# ==========================================================
# ضع الرابط المباشر اللي نسخته من Hugging Face هنا
MODEL_URL = "https://huggingface.co/ahmedhosni10/teeth-disease-model/resolve/main/teeth_best_model_vgg16.h5" 
MODEL_PATH = "teeth_best_model_vgg16.h5"
# ==========================================================

# @st.cache_resource وظيفتها إنها تمنع تحميل الموديل كل مرة التطبيق يعمل ريفرش
@st.cache_resource
def download_and_load_model(model_path, model_url):
    """
    Downloads the model from the given URL if it doesn't exist,
    then loads and returns the model.
    """
    # التأكد إذا كان ملف الموديل موجود أم لا
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... قد يستغرق هذا بعض الوقت"):
            try:
                urlretrieve(model_url, model_path)
                st.success("تم تحميل الموديل بنجاح!")
            except Exception as e:
                st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
                return None
    
    # تحميل الموديل بعد التأكد من وجوده
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"فشل تحميل الموديل من الملف: {e}")
        return None

# --- بداية منطق التطبيق بتاعك ---
st.title("🦷 Teeth Disease Classification")

# استدعاء الدالة لتحميل الموديل
model = download_and_load_model(MODEL_PATH, MODEL_URL)

# نتأكد إن الموديل تم تحميله بنجاح قبل ما نكمل
if model:
    uploaded_file = st.file_uploader("اختر صورة للأسنان...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # هنا تكمل الكود بتاعك الخاص بمعالجة الصورة وعمل الـ prediction
        st.image(uploaded_file, caption='الصورة المرفوعة', use_column_width=True)
        # ... بقية الكود بتاعك
else:
    st.error("لم يتم تحميل الموديل، لا يمكن المتابعة. يرجى مراجعة الأخطاء.")