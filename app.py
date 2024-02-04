from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

# # Load a pretrained YOLOv8n model
model = YOLO('model/bone_fraction_model.pt')

def predict_image(image):
    pred = model(image)
    st.write(pred[0].boxes.conf)
    for p in pred:
        im_array = p.plot()
        im = Image.fromarray(im_array)  
    return im

def main():
    st.title("Bone Fraction Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        # st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform prediction
        prediction_result = predict_image(np.array(image))

        # Display prediction result
        st.image(prediction_result, caption="Result", use_column_width=True)

if __name__ == "__main__":
    main()
