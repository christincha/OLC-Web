"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
from st_pages import add_page_title
import cv2
import io
import imageio
import streamlit as st
import pandas as pd
from ruamel.yaml import YAML

from training_utils.training import train_unsup_network
from utils.data_formatting import render_plotly_ui, update_state, query_data

st.set_page_config(layout="wide")
st.session_state.file_directory = None

yaml = YAML()


# Initializes the session state variables
def initialize_state():
    """
    Initializes variables in Streamlit Session State
    """
    for q in ["x-y"]:
        if f"{q}_query" not in st.session_state:
            st.session_state[f"{q}_query"] = set()
        if f"{q}-sel" not in st.session_state:
            st.session_state[f"{q}-sel"] = []

    if "counter_clustering" not in st.session_state:
        # Counter for clustering session
        st.session_state.counter_clustering = 0

    if "counter" not in st.session_state:
        # Counter for global session
        st.session_state.counter= 0

    if "but_start" not in st.session_state:
        # Sets the initial state of start button
        st.session_state.but_start = False

    if "but_continue" not in st.session_state:
        # Sets the initial state of continue training button
        st.session_state.but_continue = False

    if "but_stop" not in st.session_state:
        # Sets the initial state of stop training button
        st.session_state.but_stop = True

    if "cur_epoch" not in st.session_state:
        # Sets training epoch
        # Resets to 0, when start training clicked
        st.session_state.cur_epoch = 0

    if "training_on" not in st.session_state:
        # Sets the training state, True to train the model
        st.session_state.training_on = False

    if "click_rerun" not in st.session_state:
        # Sets the data generation state, True to regenerate plot data
        st.session_state.click_rerun = False

    if "classification_page" not in st.session_state:
        # Sets the classification state, True to enable classification page
        st.session_state.classification_page = True

initialize_state()
add_page_title()

st.write("OpenLabCluster Step 2: Generate Cluster Map")
# if st.session_state.file_directory:
#     st.write(f"File Directory: {st.session_state.file_directory}")



st.markdown("#### Set up Training Parameters")
# Divides the page into five columns
col1, col2, col3, col4, col5 = st.columns(5)

# Columns 1, 2, and 3 for entering numbers
with col1:
    num_cm_update = st.number_input("Update Cluster Map Every (Epochs)", min_value=0, max_value=1000, step=1, value=1)

with col2:
    num_cm_save  = st.number_input("Save Cluster Map Every (Epochs)", min_value=0, max_value=1000, step=1, value=5)

with col3:
    num_m_ep= st.number_input("Maximum Training Epochs", min_value=0, max_value=1000, step=1, value=10)


with col4:
    num_cm_d = st.selectbox("Cluster Map Dimension", ["2d"]) # not supporting "3d" currently
    if 'num_cm_d' not in st.session_state:
        st.session_state['num_cm_d'] = num_cm_d

with col5:
    dim_red_method = st.selectbox("Dimension Reduction Method", ["PCA", "tSNE", "UMAP"])
    if 'dim_red_method' not in st.session_state:
        st.session_state['dim_red_method'] = dim_red_method

# Creates 4 columns for buttons
col21, col22, col23, col24 = st.columns(4)
with col21:
    sc_event = st.button("Start Clustering", disabled=st.session_state.but_start)

with col22:
    stop_event = st.button("Stop Clustering", disabled=st.session_state.but_stop)

with col23:
    cc_event = st.button("Continue Clustering", disabled=st.session_state.but_continue)

with col24:
    gc_event = st.button("Go to Classification")


if st.session_state.counter_clustering == 0:
    # Initializes the training worker
    work = train_unsup_network(
        st.session_state.config_dict,
        displayiters=num_cm_update,
        saveiters=num_cm_update,
        maxiters=num_m_ep,
        reducer_name=dim_red_method,
        dimension=num_cm_d,
        continue_training=True
    )
    st.session_state.work = work
    st.session_state.counter_clustering += 1
    data = work.plot_data(st.session_state.cur_epoch)
    st.session_state.data = data
else:
    work = st.session_state.work
    data = work.plot_data(st.session_state.cur_epoch)
    st.session_state.data = data

if sc_event:
    # Re-initializes the training worker
    work = train_unsup_network(
        st.session_state.config_dict,
        displayiters=num_cm_update,
        saveiters=num_cm_update,
        maxiters=num_m_ep,
        reducer_name=dim_red_method,
        dimension=num_cm_d,
        continue_training=False
    )
    st.session_state.work = work
    st.session_state.training_on = True
    st.session_state.but_start = True
    st.session_state.but_stop = False
    st.session_state.but_continue = True
    st.session_state.counter_clustering = 1
    st.session_state.cur_epoch = 0

elif cc_event:
    # Continues clustering from earlier checkpoint
    if st.session_state.counter_clustering == 0:
        work = train_unsup_network(
            st.session_state.config_dict,
            displayiters=num_cm_update,
            saveiters=num_cm_update,
            maxiters=num_m_ep,
            reducer_name=dim_red_method,
            dimension=num_cm_d,
            continue_training=True
        )
        st.session_state.counter_clustering += 1
    st.session_state.training_on = True
    st.session_state.but_start = True
    st.session_state.but_stop = False
    st.session_state.but_continue = True
    st.session_state.cur_epoch = 0

elif stop_event:
    # Stops training
    st.session_state.training_on = False
    st.session_state.but_start = False
    st.session_state.but_stop = True
    st.session_state.but_continue = False
if gc_event:
    st.session_state.classification_page = True

if dim_red_method != st.session_state['dim_red_method'] or num_cm_d != st.session_state['num_cm_d']:
    work.update_parameters(reducer_name=dim_red_method, dimension=num_cm_d)
    data = work.plot_data(st.session_state.cur_epoch)
    st.session_state.data = data
    st.session_state['dim_red_method'] = dim_red_method
    st.session_state['num_cm_d'] =  num_cm_d
# Divides the page into two columns for scatter plot and video display
col1, col2 = st.columns(2)

# Scatter Plot
with col1:
    selected_sample = None
    st.markdown("##### Cluster Map")

    # if not st.session_state.click_rerun:
    #     if not st.session_state.training_on and not st.session_state.cur_epoch > 0:
    #         # Initializes the first session state data
    #         # Updates the data using the work if is not due to the click_rerun and training

    # Prepares data for scatter plot
    data = query_data(st.session_state.data)
    current_query, selected_sample = render_plotly_ui(data)
    rerun = update_state(current_query, data)
    if rerun and not st.session_state.training_on:
        # Reruns to update the selected point
        st.session_state.click_rerun = True
        st.experimental_rerun()

    if st.session_state.training_on:
        # Trains the model when training_on is set true
        # Starts from current epoch
        start = st.session_state.cur_epoch
        print('event sc', sc_event, cc_event, start, st.session_state.cur_epoch)
        for ith_epoch in range(start, num_m_ep):
            # TrainS one epoch
            data_epoch = work.train_step(ith_epoch)
            # UpdateS the global epoch
            st.session_state.cur_epoch = ith_epoch+1
            if isinstance(data_epoch, pd.DataFrame):
                st.write(f'training epoch: {ith_epoch}')
                st.session_state.data = data_epoch
                st.experimental_rerun() # RerunS so the plot will use the current data
        # Sets training on to False after finish the training
        st.session_state.training_on = False
        st.session_state.but_start = False
        st.session_state.but_stop = True
        st.session_state.but_continue = False
        st.experimental_rerun()

# Video Viewer
with col2:
    st.markdown("##### Video Viewer")
    # Gets video list
    video_file_names = st.session_state.config_dict['Project_folders']['train_videolist']
    with open(video_file_names,'r') as f:
        video_list = f.readlines()

    if selected_sample:
        #Displays video of selected sample
        print('select index',st.session_state["x-y-sel"])
        index = st.session_state["x-y-sel"][0]
        path = video_list[index].replace('\n', '').encode('ascii', 'ignore').decode('utf-8')
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        imgs = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                imgs.append(frame)
            else:
                break

        # Displays the selected video
        gif_bytesio = io.BytesIO()
        kargs = { 'duration': len(imgs)/fps }
        imageio.mimsave(gif_bytesio, imgs, 'GIF', **kargs)
        st.image(gif_bytesio.getvalue(), output_format='GIF')


