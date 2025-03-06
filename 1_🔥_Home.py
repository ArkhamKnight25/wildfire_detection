import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os
from glob import glob
from numpy import random
import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )

    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}

    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'

        if v > 1:
            prediction_text += 's'

        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    latency = sum(res[0].speed.values())
    latency = round(latency / 1000, 2)
    prediction_text += f' in {latency} seconds.'

    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text

def main():
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS for the background GIF
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://media.tenor.com/UPltgIK-cU8AAAAC/fire-fireball.gif");
            background-size: cover;
            background-position: center;
        }
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 35px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # App title
    st.markdown("<div class='title'>अग्नि neosis</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        logos = glob('dalle-logos/*.png')
        logo = random.choice(logos)
        st.image(logo, use_column_width=True)
    with col3:
        st.write("")

    st.sidebar.image(logo, use_column_width=True)

    # Add a section divider
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select Model Type",
                              ("Fire Detection", "General"), index=0)

    # Modify the model directory and allow only "fire_l" for Fire Detection
    if model_type == "Fire Detection":
        models_dir = "fire-models"
        model_files = ['fire_l']  # Only show the "fire_l" model
    else:
        models_dir = "general-models"
        model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]

    with col2:
        selected_model = st.selectbox(
            "Select Model Size", sorted(model_files), index=0)  # Default to "fire_l" (index 0)

    model_path = os.path.join(models_dir, selected_model + ".pt")
    model = load_model(model_path)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col2:
        conf_threshold = st.slider(
            "Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
    with col1:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")

    image = None
    image_source = st.radio("Select image source:",
                            ("Enter URL", "Upload from Computer"))
    if image_source == "Upload from Computer":
        uploaded_file = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Error loading image from URL.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error loading image from URL: {e}")

    if image:
        with st.spinner("Detecting"):
            prediction, text = predict_image(
                model, image, conf_threshold, iou_threshold)
            st.image(prediction, caption="Prediction", use_column_width=True)
            st.success(text)

        prediction = Image.fromarray(prediction)

        image_buffer = io.BytesIO()
        prediction.save(image_buffer, format='PNG')

        st.download_button(
            label='Download Prediction',
            data=image_buffer.getvalue(),
            file_name='prediction.png',
            mime='image/png'
        )

if __name__ == "__main__":
    main()
