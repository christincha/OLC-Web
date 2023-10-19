"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import streamlit as st
from sub_pages.page import Page
from utils.config_utils import load_cfg_file

class Load_Project(Page):
    '''
    Class to load project
    '''
    def __init__(self, **kwargs):
        name = "Load Project"
        super().__init__(name, **kwargs)
        self.cur_project_path = None

    def content(self):

        st.markdown("Load Project")
        # Uploads config file
        file = st.file_uploader("Upload the project config file")
        if file:
            # Loads config file and save current project path
            load_cfg_file(file)
            self.cur_project_path = st.session_state.config_dict['Project_folders']['project_path']
            st.write('Project Directory', self.cur_project_path)

    def file_dir(self):
        return self.cur_project_path