"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import io
import os

import cv2
import numpy as np
import pandas as pd
import imageio
import streamlit as st
from st_pages import add_page_title

from utils.data_formatting_classification import capture_plot_event, get_prev_samp, update_state, data_formating, get_cur_samp, annot
from training_utils.itertrain import train_iter_network

st.set_page_config(layout="wide")
st.session_state.file_directory = None

def initialize_state():
    """
    Initializes variables in Streamlit Session State
    """
    for q in ["index_str"]:
        if f"{q}_query" not in st.session_state:
            st.session_state[f"{q}_query"] = set()
        # Stores samples selected by AL methods
        if f"{q}_al_sel" not in st.session_state:
            st.session_state[f"{q}_al_sel"] = set()
        # Stores samples already annotated
        if f"{q}_annt" not in st.session_state:
            st.session_state[f"{q}_annt"] = set()
        # Stores samples of current annotation
        if f"{q}_cur" not in st.session_state:
            st.session_state[f"{q}_cur"] =  []

    if "counter" not in st.session_state:
        # Counter for global session
        st.session_state.counter = 0

    if "counter_video" not in st.session_state:
        # Counter for video display
        st.session_state.counter_video = False

    if "counter_cla" not in st.session_state:
        # Counter for classification session
        st.session_state.counter_cla = 0

    if "index_plt" not in st.session_state:
        # Variable for the index of video displaying
        st.session_state.index_plt = None

    if "but_start_cla" not in st.session_state:
        # Sets the initial state of start classification training button
        st.session_state.but_start_cla = False

    if "but_continue_cla" not in st.session_state:
        # Sets the initial state of continue classification training button
        st.session_state.but_continue_cla = False

    if "but_stop_cla" not in st.session_state:
        # SetS the initial state of stop classification training button
        st.session_state.but_stop_cla = True

    if "cur_epoch_cla" not in st.session_state:
        # Sets the stat epoch of classification training
        # Resets to 0 at the beginning of training
        st.session_state.cur_epoch_cla = 0

    if "training_on_cla" not in st.session_state:
        # Sets the training state, True to train the model
        st.session_state.training_on_cla = False

    if "click_rerun_cla" not in st.session_state:
        # Reruns the session when clicked
        st.session_state.click_rerun_cla = False

    if "annt_cache" not in st.session_state:
        # Variable to store annotated samples
        st.session_state.annt_cache = []

    if 'video_rand_seed' not in st.session_state:
        # Sets the start frame of video
        st.session_state.video_rand_seed = 0

add_page_title()
initialize_state()

st.write("OpenLabCluster Step 3: Generate Behavior Classification Map")

# Shows classification page when user selected go to classification
if 'classification_page' in st.session_state and st.session_state['classification_page']:
    st.markdown("#### Set up Training Parameters")
    # Divides the page into five columns
    col1, col2, col3, col4= st.columns(4)

    # Sets Parameters
    with col1:
        al_method = st.selectbox("Active Learning Method",  ["Marginal Index (MI)", "Core Set (CS)", "Cluster Center (Top)",
                            "Cluster Random (Rand)", "Uniform (Uni)", ])
        if 'al_method' not in st.session_state:
            st.session_state['al_method'] = al_method

    with col2:
        num_per_sel = st.number_input("# Samples Per Selection ", min_value=0, max_value=1000, step=1,
                                        value=10)
        if "num_per_sel"  not in st.session_state:
            st.session_state['num_per_sel'] = num_per_sel

    with col3:
        num_m_ep =  st.number_input("Maximum Training Epochs", min_value=0, max_value=1000, step=1,
                                        value=10)


    with col4:
        dim_red_method_cla = st.selectbox("Dimension Reduction Method", ["PCA", "tSNE", "UMAP"])
        if 'dim_red_method_cla' not in st.session_state:
            st.session_state['dim_red_method_cla'] = dim_red_method_cla
    # Create buttons for starting/end classification training
    col21, col22, col23, col24 = st.columns(4)
    with col21:
        sc_event = st.button("Run Classification", disabled=st.session_state.but_start_cla)

    with col22:
        stop_event = st.button("Stop Classification", disabled=st.session_state.but_stop_cla)

    with col23:
        cc_event = st.button("Next Selection", disabled=st.session_state.but_continue_cla)

    with col24:
        sr_event = st.button("Save Results")

    if st.session_state.counter_cla == 0:
        # Initializes training worker
        try:
            work = train_iter_network(
                config=st.session_state.config_dict, sample_method=al_method, num_sample=num_per_sel,
                reducer_name=dim_red_method_cla
            )
            st.session_state.work_cla = work
            st.session_state.counter_cla = 1
        except:

            st.error('Make sure the project is correctly initialized and you have performed "Start Clustering" on Cluster Map page.')

    else:
        work = st.session_state.work_cla

    if sc_event:
        # Re-initializes the training worker
        work = train_iter_network(
            config=st.session_state.config_dict, sample_method=al_method,
            num_sample=num_per_sel, reducer_name=dim_red_method_cla)
        st.session_state.work_cla = work
        st.session_state.training_on_cla = True
        st.session_state.but_start_cla = True
        st.session_state.but_stop_cla = False
        st.session_state.but_continue_cla = True
        st.session_state.counter_cla = 1
        st.session_state.cur_epoch_cla = 0
    elif cc_event:
        # Updates the training worker
        if st.session_state.counter_cla == 0:
            work = train_iter_network(
                st.session_state.config_dict, sample_method=al_method,
                num_sample=num_per_sel, reducer_name=dim_red_method_cla
            )

            st.session_state.counter_cla = 1
        else:
            st.session_state.work_cla.update_dataloader()
        st.session_state.training_on_cla = True
        st.session_state.but_start_cla = True
        st.session_state.but_stop_cla = False
        st.session_state.but_continue_cla = True
        st.session_state.cur_epoch_cla = 0

    elif stop_event:
        # Updates button after stop training
        st.session_state.training_on_cla = False
        st.session_state.but_start_cla = False
        st.session_state.but_stop_cla = True
        st.session_state.but_continue_cla = False

    # Updates model parameter
    if al_method != st.session_state['al_method'] or num_per_sel !=st.session_state['num_per_sel']:
        work.update_parameters(sample_method=al_method, num_sample=num_per_sel, reducer_name=dim_red_method_cla)
        st.session_state['al_method'] = al_method
        st.session_state['num_per_sel'] = num_per_sel


    # Divides the page into three columns for scatter plot, video displaying, and annotating
    plot_col1, plot_col2, plot_col3 = st.columns([0.4,0.4,0.2])
    with plot_col1:
        st.markdown("##### Behavior Classification Map")
        if 'data_cla' not in st.session_state or dim_red_method_cla != st.session_state['dim_red_method_cla']:
            # Initializes the first session state data
            # Updates the data using the work if is not due to the click_rerun and training
            # Gets the id of previous labeled data
            work.update_parameters(sample_method=al_method, num_sample=num_per_sel, reducer_name=dim_red_method_cla)
            st.session_state['dim_red_method_cla'] = dim_red_method_cla
            data, semi_label = work.plot_data(st.session_state.cur_epoch_cla)
            st.session_state.semi_label = semi_label
            if sum(semi_label!=0) > 0:
                labeled_id = list(np.where(semi_label!=0)[0])
                st.session_state[f"index_str_annt"] = st.session_state[f"index_str_annt"] | set(labeled_id)
            # Initializes the set for annotation
            new_tolabel = work.active_label_selection(st.session_state.cur_epoch_cla)
            st.session_state["index_str_al_sel"] =  set(new_tolabel)
            st.session_state.data_cla = data

        # Plots with the stored st.session_state.data_cla
        data = data_formating(st.session_state.data_cla)
        current_query, selected_sample = capture_plot_event(data)
        rerun = update_state(current_query, data)

        # Reruns to update the selected point when not training the model
        if rerun and not st.session_state.training_on_cla:
            print('set rerun here')
            st.session_state.click_rerun_cla = True
            st.experimental_rerun()

        # Trains the model when training_on is set true
        if st.session_state.training_on_cla:

            start = st.session_state.cur_epoch_cla
            print('event sc', sc_event, cc_event, start, st.session_state.cur_epoch_cla)

            for ith_epoch in range(start, num_m_ep):
                print('training epoch', ith_epoch, 'max num epoch', num_m_ep)
                data_epoch, semi_label  = work.train_step(ith_epoch)
                # Updates the global epoch
                st.session_state.cur_epoch_cla = ith_epoch + 1
                if isinstance(data_epoch, pd.DataFrame):
                    st.write(f'training epoch: {ith_epoch}')
                    st.session_state.data_cla = data_epoch
                    st.experimental_rerun()

            # Selects a set of label when finished the training
            new_tolabel = work.active_label_selection(st.session_state.cur_epoch_cla)
            st.session_state["index_str_al_sel"] =  set(new_tolabel)
            # Sets training on to False after finish the training
            st.session_state.training_on_cla = False
            st.session_state.but_start_cla = False
            st.session_state.but_stop_cla = True
            st.session_state.but_continue_cla = False
            st.experimental_rerun()

    # Page 2: Video Viewer

    with plot_col2:
        st.markdown("##### Video Viewer")
        butcol1, butcol2, butcol3 = st.columns([0.43,0.24,0.33])
        with butcol1:
            prev_button = st.button("<<Previous")
        with butcol2:
            play_button = st.button('Play' )
        with butcol3:
            next_button = st.button("Next>>")

        video_file_names = st.session_state.config_dict['Project_folders']['train_videolist']#'/home/ws2/Documents/jingyuan/OpenLabCluster/openlabcluster/gui/test2-2022-06-14/videos/video_segments_names.text'
        with open(video_file_names, 'r') as f:
            video_list = f.readlines()

        # Get the current sample when click the play button
        if play_button:
            st.session_state.index_plt = get_cur_samp()
            st.session_state.counter_video = True
            print('session state index', st.session_state.index_plt, 'selected',st.session_state["index_str_al_sel"])
            st.experimental_rerun()
        if next_button:
            print('index in page', st.session_state.index_plt, 'selected',st.session_state["index_str_al_sel"])
            index = annot()

            if index is not None:
                st.session_state.index_plt = index
                label_path = st.session_state.config_dict['Project_folders']['label_path']
                np.save(os.path.join(label_path, 'label.npy'), st.session_state.semi_label)
                st.experimental_rerun()
                st.session_state.counter_video = True
            else:
                st.write('End of Selected Samples')
                st.session_state.counter_video = False

        if prev_button:
            index = get_prev_samp()
            if index==None :
                st.write('No Previous Video')
                st.session_state.counter_video = False
            else:
                st.session_state.index_plt = index
                st.experimental_rerun()

        if st.session_state.counter_video:
            # Loads video and displays
            path = video_list[st.session_state.index_plt].replace('\n', '').encode('ascii', 'ignore').decode('utf-8')
            print('video path', path)
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            imgs = []
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    imgs.append(frame)
                else:
                    break
            cap.release()
            gif_bytesio = io.BytesIO()

            # Change the start point, solve the gif is not played as animate for the same input
            st.session_state.video_rand_seed = -1*st.session_state.video_rand_seed +1
            kargs = { 'duration': len(imgs)/fps }
            imageio.mimsave(gif_bytesio, imgs[st.session_state.video_rand_seed:], 'GIF', **kargs)

            # Displays the GIF animation
            st.image(gif_bytesio.getvalue(), output_format='GIF' )
            gif_bytesio.close()

    with plot_col3:
        class_list = st.session_state.config_dict['Model']['class_name']
            # ["unkown","drink", "eat", "groom", "head","rear", "rest", "walk",
            #            "hang"]
        class_array = np.asarray(class_list)
        class_label = st.radio(
            "Class Name",
           class_list)
        cla_id =  np.where(class_array == class_label)[0]
        if st.session_state.counter_video:
            st.session_state.semi_label[st.session_state.index_plt] = cla_id

        st.markdown(
            """<style>
        div[class*="stRadio"] > options[0] > div[data-testid="element-container"] > p {
            font-size: 11px;
        }
            </style>
            """, unsafe_allow_html=True)

