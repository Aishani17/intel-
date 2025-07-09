%%writefile app.py
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# --- Page Configuration ---
st.set_page_config(page_title="KD Demo: Teacher vs Student", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #4a90e2;
            text-align: center;
        }
        .subheader {
            color: #555;
            font-size: 20px;
            text-align: center;
            margin-top: -10px;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            color: #999;
            margin-top: 30px;
        }
        .uploaded-image {
            border-radius: 10px;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title and Subtitle ---
st.markdown('<div class="title">🧠 Knowledge Distillation Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Compare Outputs of Teacher and Student Models on Your Uploaded Image</div>', unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    teacher = load_model("https://drive.google.com/file/d/1-09RblgTlvNECPI5AZet29XmJphF7aHr/view?usp=sharing", compile=False)
    student = load_model("https://drive.google.com/file/d/1BlzHIVppHi50eG3oTLMloQnsrkET2JEx/view?usp=sharing", compile=False)
    return teacher, student

# --- Image Pre/Post Processing ---
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize as per model input
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def postprocess_image(pred):
    pred = np.clip(pred[0], 0, 1) * 255
    return Image.fromarray(pred.astype('uint8'))

# --- Load Models ---
teacher_model, student_model = load_models()

# --- Upload Interface ---
uploaded_file = st.file_uploader("📁 Upload an image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Original Image", use_column_width=True, output_format="PNG")

    input_data = preprocess_image(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👨‍🏫 Teacher Model Output")
        teacher_output = teacher_model.predict(input_data)
        teacher_result = postprocess_image(teacher_output)
        st.image(teacher_result, use_column_width=True)

    with col2:
        st.subheader("🧑‍🎓 Student Model Output")
        student_output = student_model.predict(input_data)
        student_result = postprocess_image(student_output)
        st.image(student_result, use_column_width=True)

# --- Footer ---
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
