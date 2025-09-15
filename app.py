import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download  # <--- 1. استيراد المكتبة الجديدة
import os

# --- معلومات الموديل من Hugging Face ---
REPO_ID = "https://huggingface.co/ahmedhosni10/teeth-disease-model/resolve/main/teeth_best_model_vgg16.h5"  # <--- 2. اسم المستودع بتاعك
FILENAME = "teeth_best_model_vgg16.h5"     # <--- 3. اسم ملف الموديل

@st.cache_resource
def load_model_from_hf(repo_id, filename):
    """
    Downloads the model from Hugging Face Hub and caches it.
    """
    with st.spinner("Downloading model... قد يستغرق هذا بعض الوقت"):
        try:
            # استخدام الدالة الرسمية للتحميل
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            # تحميل الموديل بعد تنزيله
            model = tf.keras.models.load_model(model_path)
            st.success("تم تحميل الموديل بنجاح!")
            return model
        except Exception as e:
            st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
            return None

# --- بداية منطق التطبيق ---
st.title("🦷 Teeth Disease Classification")

# استدعاء الدالة لتحميل الموديل
model = load_model_from_hf(REPO_ID, FILENAME)

if model:
    uploaded_file = st.file_uploader("اختر صورة للأسنان...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # هنا تكمل الكود بتاعك الخاص بمعالجة الصورة وعمل الـ prediction
        st.image(uploaded_file, caption='الصورة المرفوعة', use_column_width=True)
        # ... بقية الكود بتاعك لمعالجة الصورة والتنبؤ
else:
    st.error("لم يتم تحميل الموديل، لا يمكن المتابعة. يرجى مراجعة الأخطاء.")
