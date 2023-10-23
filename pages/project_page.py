"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import streamlit as st
import numpy as np
from st_pages import add_page_title
from pages.sub_pages.load_project_page import Load_Project
from pages.sub_pages.creat_new_project import Create_New_Project

add_page_title()

st.write("OpenLabCLuster Step 1: Create a New Project or Load a Project")

PAGES = {
    "Create New Project": Create_New_Project,
    "Load Project": Load_Project
}

st.session_state.file_directory = None
# Select pages
selection = st.radio("Chose the Project Loading Method", list(PAGES.keys()))
page = PAGES[selection]
data = np.zeros(10)
with st.spinner(f"{selection} ..."):
    page = page()
    page()
    # Gets directory of current project
    st.session_state.file_directory = page.file_dir()
    print(st.session_state.file_directory)