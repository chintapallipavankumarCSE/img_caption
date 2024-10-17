# Import necessary libraries
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import requests

# Load the BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Function to generate caption from image
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Set the title
st.title("Image Captioner")

# Create an input box for the image URL
st.write("Enter Image URL")
image_url = st.text_input("")

# OR divider
st.write("OR")

# File uploader for image upload
st.write("Upload Image")
uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png", "jpeg"])

# Button to process the image
if st.button("Generate Caption"):
    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Generate caption using BLIP model
        caption = generate_caption(image)
        st.write("Generated Caption:", caption)

    elif image_url:
        try:
            # Load the image from the URL
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            st.image(image, caption="Image from URL", use_column_width=True)
            
            # Generate caption using BLIP model
            caption = generate_caption(image)
            st.write("Generated Caption:", caption)
        except Exception as e:
            st.write("Error loading image from URL. Please check the URL.")
    else:
        st.write("Please upload an image or provide an image URL.")
