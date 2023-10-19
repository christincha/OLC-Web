"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import streamlit as st

class Page:
    # Class template for load project page
    def __init__(self, name,  **kwargs):
        self.name = name
        self.kwargs = kwargs

    def content(self):
        """Returns the content of the page"""
        raise NotImplementedError("Please implement this method.")

    def title(self):
        """Returns the title of the page"""
        st.header(f"{self.name}")

    def __call__(self):
        self.title()
        self.content()