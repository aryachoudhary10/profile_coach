import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# --- MODEL LOADING (Done once at the start) ---
# This is the "magic" model from Hugging Face that describes images.
# It will be downloaded automatically the first time you run it.
@st.cache_resource
def load_captioner():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

captioner = load_captioner()

# OpenCV's pre-trained face detector
@st.cache_resource
def load_face_detector():
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    # You need to download these two files first. A quick google search will find them.
    # For convenience, I'd just download them from the OpenCV GitHub repo.
    # Let's assume you have them in your project folder.
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return net
    except cv2.error:
        st.error("Face detector model files not found! Please download 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel'.")
        return None

face_detector = load_face_detector()


# --- ANALYSIS FUNCTIONS ---

def analyze_face(image_np):
    """Analyzes the number of faces in the image."""
    if face_detector is None:
        return "Face detector not loaded.", "neutral"
    
    (h, w) = image_np.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    face_count = 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            face_count += 1
            
    if face_count == 0:
        return "No clear face detected. Try a photo where your face is visible.", "bad"
    elif face_count == 1:
        return "Great! A clear solo photo works best.", "good"
    else:
        return f"Group photo detected ({face_count} people). For a main profile picture, a solo photo is usually better.", "warning"

def analyze_blur(image_np):
    """Analyzes the blurriness of the image."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if fm < 100:  # Threshold for blurriness
        return f"The image appears blurry (Score: {fm:.2f}). Use a sharper photo.", "bad"
    else:
        return f"Nice and sharp! (Score: {fm:.2f})", "good"

def get_caption(pil_image):
    """Generates a caption for the image."""
    caption_result = captioner(pil_image)
    return f"This photo shows: \"{caption_result[0]['generated_text']}\"", "neutral"


# --- STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🤖 AI Profile Photo Coach")
st.write("Upload a photo to get instant, AI-powered feedback. Let's find your best shot!")

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    # Convert RGB to BGR for OpenCV
    image_np_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Your Uploaded Photo", use_column_width=True)

    with col2:
        st.subheader("Analysis & Feedback:")
        
        # --- Run Analyses and Display Results ---
        # Face Analysis
        face_feedback, face_status = analyze_face(image_np_cv)
        if face_status == "good":
            st.success(f"✅ **Face Detection:** {face_feedback}")
        elif face_status == "warning":
            st.warning(f"⚠️ **Face Detection:** {face_feedback}")
        else:
            st.error(f"❌ **Face Detection:** {face_feedback}")

        # Blur Analysis
        blur_feedback, blur_status = analyze_blur(image_np_cv)
        if blur_status == "good":
            st.success(f"✅ **Image Quality:** {blur_feedback}")
        else:
            st.error(f"❌ **Image Quality:** {blur_feedback}")
        
        # Content Analysis
        caption_feedback, _ = get_caption(image)
        st.info(f"💡 **Content Analysis:** {caption_feedback}")