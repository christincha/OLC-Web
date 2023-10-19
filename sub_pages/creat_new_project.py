"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import os
import  datetime
import streamlit as st

from sub_pages.page import Page
from utils.config_utils import get_default_config

class Create_New_Project(Page):
    def __init__(self, **kwargs):
        """ Creates new project
        """
        name = "Create New Project"
        super().__init__(name,  **kwargs)
        self.cur_project_path = None


    def content(self):
        "Sets the name for new project"
        project_name = st.text_input("Enter Project Name")

        # Sets default root of directory
        home_directory = os.path.expanduser("~")
        project_directory = os.path.join(home_directory, 'OLC_Project')
        if not os.path.exists(project_directory):
            os.mkdir(project_directory)
        cur_time = datetime.datetime.now()

        # Creates curren project
        if project_name:
            cur_project_path = os.path.join(project_directory, f'{project_name}-{cur_time.year}-{cur_time.day}-{cur_time.month}')
            st.session_state.project_path = cur_project_path
            if not os.path.exists(cur_project_path):
                os.mkdir((cur_project_path))

            # Updates the current project path
            if not self.cur_project_path:
                self.cur_project_path = cur_project_path
            if project_name:
                st.write(f"Project Name: {project_name}")
                st.write(f"Project Default Directory: {cur_project_path}")

            # User input for dataset root path and dataset filename
            data_folder_path = st.text_input('Enter the Directory to Keypoints or Precomputed Kinematics (h5 file)')
            data_file_name = st.text_input('Enter the Filename of Keypoints or Precomputed Kinematics (h5 file)')
            file_video_list = st.file_uploader("Load Video Segments Names List")

            if file_video_list:
                # Saves video name list to current folder
                st.write(f"Selected File: {file_video_list.name}")
                file_video_dir = os.path.join(cur_project_path, 'videos', 'video_segments_names.text')
                with open(file_video_dir, 'wb') as f:
                    f.write(file_video_list.getbuffer())

            # Option to use gpu
            gpu_state = st.checkbox("USE GPU")
            if gpu_state:
                device  = 'cuda'
            else:
                device = 'cpu'
            # User input for feature length
            feature_length = st.number_input("Feature Length (int)", min_value=0, max_value=10000, value=1, step = 1)

            # User option for the active learning method
            al_state = st.checkbox("Advanced Options")
            sample_method = "Marginal Index (MI)"
            if al_state:
                option1 = st.selectbox("Active Learning Methods", ["Marginal Index (MI)", "Core Set (CS)", "Cluster Center (Top)", "Uniform (Random)"])
                sample_method = option1

            # Creates config file
            config_kwargs = {'device': device, 'sample_method': sample_method, 'feature_length': feature_length,
                             'train':data_file_name, 'data_path':data_folder_path}
            get_default_config(st.session_state.project_path, project_name, **config_kwargs)

    def file_dir(self):
        return self.cur_project_path