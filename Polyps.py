import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Define the model path
model_path = 'best.pt'

# Initialize the YOLOv8 model
model = YOLO(model_path)

# Streamlit UI
st.title('YOLOv8 Polyps Detection')

st.sidebar.title('Image Config')
# Set the default confidence to a higher value, such as 0.5
confidence = st.sidebar.slider('Confidence', min_value=0.00, max_value=0.10, value=0.01)
source_img = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])

st.title("Object Detection")
st.caption('Upload a photo and click the Detect Objects button to see the results.')

# Display the uploaded image
if source_img:
    uploaded_image = Image.open(source_img)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button('Detect Objects'):
        # Convert the image to a format suitable for OpenCV
        image_np = np.array(uploaded_image.convert("RGB"))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform inference with YOLOv8
        results = model.predict(image_np, conf=confidence)

        if len(results[0].boxes) == 0:
            st.write("No objects detected.")
        else:
            annotated_image = results[0].plot()

            # Display the annotated image using Streamlit
            st.image(annotated_image[:, :, ::-1], caption='Detected Image', use_column_width=True)

            # Display detection results
            with st.expander("Detection Results"):
                for box in results[0].boxes:
                    st.write(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xywh}")

else:
    st.info("Please upload an image to detect objects.")
