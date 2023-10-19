"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
from pathlib import Path
import streamlit as st
from st_pages import Page, Section, add_page_title, show_pages

# Initializes Pages
st.set_page_config(layout="wide")
show_pages(
    [Page("main_page.py", "OpenLabCluster Home Page", "🏠"),
     Page("pages/project_page.py", "Project Manager"),
     Section("Model Learner"),
     Page("pages/clustering_page.py", "Cluster Map"),
     Page("pages/classification_page.py", "Behavior Classification Map")
     ]
)
add_page_title()

# Shows Logo
st.image('./logo/GUIplot.png')
with st.expander("Show documentation"):
    # Adds documentation to main page
    intro_markdown = Path("README.md").read_text()
    st.markdown(intro_markdown, unsafe_allow_html=True)
