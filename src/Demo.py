
import streamlit as st
from PIL import Image
import tqdm
import glob
import cv2
from matplotlib import pyplot as plt
# Local imports
from query_processing import run_query_inference_only
channel_bins = 16
experimentation = True
display_results = True
dataset_path = "/Users/ameya/Documents/DIP/dataset/training_set/"

def display_images_with_labels(images, labels):
    for i, (image_path, label) in enumerate(zip(images, labels)):
        st.subheader(f"Result {i + 1}")
        
        # Display image
        image = Image.open(image_path)
        st.image(image)

        # Display label
        st.write(f"Label {i + 1}: {label.capitalize()}")
        st.write("------")

def perfrom_cbir(dataset_path, query_image_path, distribution,
                            proximity, channel_bins, k, experimentation, display_results):
    # Run the query and get results
    images, labels = run_query_inference_only(dataset_path, query_image_path, distribution, proximity,
                                 channel_bins, k, display_results=display_results, experimentation = False)
    return images, labels

def main():
    st.title("Content Based Image Retrival")

    # CSS styling for the title
    st.markdown(
        """
        <style>
            .title {
                color: #5A5A5A;
                text-align: center;
                padding: 10px;
                font-size: 24px;
                background-color: #EFEFEF;
                border-radius: 10px;
                margin-bottom: 20px;
            }

            .dropdown:hover {
                transform: scale(1.05);
            }

            .styled-text {
                font-size: 16px;
                color: #333;
                cursor: pointer;
                transition: color 0.3s ease-in-out;
            }

            .styled-text:hover {
                color: #FF5733;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add a styled title
    st.markdown("<p class='title'>Choose an image and set the Parameters</p>", unsafe_allow_html=True)

    query_image = st.file_uploader("Upload Image", type=["JPG", "JPEG", "PNG"])

    if query_image is not None:
        st.image(query_image, caption="Uploaded Image", use_column_width=True)
        file_extension = query_image.name.split(".")[-1]
        query_image_path = f"./temp_query_image.{file_extension.lower()}"  # Lowercase extension to handle variations

        with open(query_image_path, "wb") as f:
            f.write(query_image.read())

        # Apply hover zoom effect to the dropdown
        distribution = st.selectbox(
            "Select Frequency Distribution",
            ['Basic-Histogram', 'Split-histogram', 'CCV'],
            index=0,
            key="dropdown",
            help="Choose the Frequency Distribution"
        )

        # Apply hover effect to styled text
        proximity = st.selectbox(
            "Select Proximity Measure",
            ['l1', 'l2', 'corr'],
            index=2,
            key="styled-text",
            help="Choose the proximity measure"
        )

        k = st.number_input("Enter the value of k", min_value=1, value=5)
        if st.button("Go"):
          
            images, labels = perfrom_cbir(dataset_path = dataset_path, query_image_path = query_image_path,
                 distribution = distribution, proximity = proximity, channel_bins = channel_bins,
                 k = k, experimentation = experimentation, display_results = display_results)
            
            
            display_images_with_labels(images, labels)

    st.markdown("<p class='fadeIn' style='text-align: center; margin-top: 50px;'>Made by Ameya and Dhruv</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

