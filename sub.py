import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io

# Mock model function
def detect_changes(image1, image2):
    # Placeholder function for change detection
    return np.random.rand()  # Random similarity score for demonstration

# Title and description
st.title("Satellite Imagery Change Detection")
st.write("Upload two satellite images to detect changes.")

# Image uploaders
uploaded_image1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"])
uploaded_image2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"])

if uploaded_image1 and uploaded_image2:
    image1 = Image.open(uploaded_image1)
    image2 = Image.open(uploaded_image2)

    # Display images
    st.image(image1, caption="Image 1", use_column_width=True)
    st.image(image2, caption="Image 2", use_column_width=True)

    # Convert images to bytes
    image1_bytes = io.BytesIO()
    image1.save(image1_bytes, format='PNG')
    image1_bytes.seek(0)
    
    image2_bytes = io.BytesIO()
    image2.save(image2_bytes, format='PNG')
    image2_bytes.seek(0)
    
    # Perform change detection
    change_score = detect_changes(image1_bytes, image2_bytes)
    
    # Display result
    st.write(f"Change Detection Score: {change_score:.2f}")

    # Optional: Display a plot (e.g., histogram of changes)
    fig, ax = plt.subplots()
    ax.hist([change_score], bins=10, alpha=0.7)
    ax.set_title("Change Detection Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
