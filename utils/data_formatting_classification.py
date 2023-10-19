import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict
from typing import Set
from streamlit_plotly_events import plotly_events
import plotly.express as px
import plotly.graph_objs as go

def data_formating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the dataset frame including color information for scatter plot
    Inputs:
        df: projected latent representation for scatter plot in pandas data format
    """
    # Saves sample index
    df["index_str"] = (
        df["index"].astype(int)
    )
    df["selected"] = "gray" # -1 means not selected/ not annotated /not current
    # Saves the x-y for plotly click event
    df["x-y"] = (
        (1000 * df["x"]).astype(int).astype(str)
        + "-"
        + (1000 * df["y"]).astype(int).astype(str)
    )
    for q in ["index_str"]:
        # Updates color setting
        # AL selected samples -> blue, annotated samples -> green, current sample -> red
        if st.session_state[f"{q}_al_sel"]:
            df.loc[df[q].isin(st.session_state[f"{q}_al_sel"]), "selected"] = "blue" # selected as 0
        if st.session_state[f"{q}_annt"]:
            df.loc[df[q].isin(st.session_state[f"{q}_annt"]), "selected"] = "green"  # selected as 0
        if st.session_state[f"{q}_cur"]:
            df.loc[df[q].isin(st.session_state[f"{q}_cur"]), "selected"] = "red" # selected as 0

    return df

def update_state(current_query: Dict[str, Set], df):
    """
    Updates session state based on the current query
    Inputs:
        current_query: user selected sample
        df: projected latent representation in pandas data format
    """
    rerun = False
    for q in ["index_str"]:
        # Updates the state when current query is not in session state
        if current_query[f"{q}_query"] - st.session_state[f"{q}_query"]:
            st.session_state[f"{q}_query"] = current_query[f"{q}_query"]
            # print('current query',list(current_query[f"{q}_query"]))
            # print('x-y', df["x-y"], list(current_query[f"{q}_query"])[0])
            # Gets index of current sample
            cur_index = np.where(df["x-y"] == list(current_query[f"{q}_query"])[0])
            print(cur_index)
            if len(cur_index[0]) > 0:
                cur_index = cur_index[0][0] #df.loc[0]["x-y"])[0][0]#list(current_query[f"{q}_query"])[0])[0]
                print(cur_index, df.loc[cur_index]["x-y"])
                if cur_index not in st.session_state[f"{q}_al_sel"]:
                    # Includes the sample index for annotation
                    st.session_state[f"{q}_al_sel"].add(cur_index)
                else:
                    # Excludes the sample for annotation
                    st.session_state[f"{q}_al_sel"].remove(cur_index)
            rerun = True
    return rerun


def plot_latent_representation(df: pd.DataFrame) -> go.Figure:
    """
    Plots the projected latent representation
    Inputs:
        df: projected latent representation in pandas data format
    """
    fig = px.scatter(
        df,
        "x",
        "y",
        color="selected",
        color_discrete_sequence=["gray", "blue", "green", "red"],
        category_orders={"selected": ["gray", "blue", "green", "red"]},
        hover_data=[
            "x",
            "y",
        ],
        width=300, height=300
    )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig['layout'].update(margin=dict(l=0, r=0, b=0, t=0))
    return fig


def capture_plot_event(transformed_df: pd.DataFrame) -> tuple:
    """
    Enables the click event of plotly
    Input:
        Projected latent representation in pandas data format
    """
    #c1, c2 = st.columns(2)

    bill_to_tip_figure = plot_latent_representation(transformed_df)
    sample_selected = plotly_events(
        bill_to_tip_figure,
        click_event=True, hover_event=False,
        key=f"index_str_{st.session_state.counter}",
    )
    # if sample_selected:
    #     transformed_df.loc[sample_selected[0]['pointIndex']]['selected'] = True
    print(sample_selected)
    # with c2:
    #     size_to_time_clicked = plotly_events(
    #         size_to_time_figure,
    #         click_event=True,
    #         key=f"size_to_time_{st.session_state.counter}",
    #     )
    #     day_clicked = plotly_events(
    #         day_figure,
    #         click_event=True,
    #         key=f"day_{st.session_state.counter}",
    #     )

    current_query = {}

    current_query["index_str_query"] =   {
        f"{int(1000*el['x'])}-{int(1000*el['y'])}" for el in sample_selected[:1]
    }
    return current_query, sample_selected

def annot():
    # Updates the info of the annotated sample
    # print(cur_index, st.session_state['index_str_cur'])
    # assert cur_index in st.session_state['index_str_cur']

    # Remove the current sample from the al selection set and the current annotation set

    #st.session_state['index_str_cur'].remove(cur_index)
    # Adds the current sample to annotated set
    if len(st.session_state['index_str_cur']) > 0:
        cur_index = st.session_state['index_str_cur'][0]
        st.session_state['index_str_annt'].add(cur_index)
        st.session_state['annt_cache'].append(cur_index)

    # chooses one from the al_sel as current for annotation
    if len(st.session_state['index_str_al_sel']) >0:
        next = int(list(st.session_state['index_str_al_sel'])[0])
        st.session_state['index_str_cur'] = [next]
        st.session_state['index_str_al_sel'].remove(next)
    else:
        next = None
        st.session_state['index_str_cur'] = []
    return next

def get_cur_samp():
    # Returns index of current sample
    if st.session_state['index_str_cur']:
        return int(st.session_state['index_str_cur'][0])
    else:
        next = list(st.session_state['index_str_al_sel'])[0]
        st.session_state['index_str_cur'] = [next]
        st.session_state['index_str_al_sel'].remove(next)
        return int(next)

def get_prev_samp():
    # Returns the ID of previous selected sample
    print('annotate cache', st.session_state['annt_cache'])
    if len(st.session_state['annt_cache']) and len(st.session_state['index_str_cur']> 0):
        cur_index =  int(st.session_state['index_str_cur'][0])
        prev_index = st.session_state['annt_cache'][-1]
        # Replaces prev_index to the current index set
        st.session_state['index_str_cur'] =[prev_index]
        # Re-adds current sample to al selected set
        st.session_state['index_str_al_sel'].add(cur_index)
        # Updates annt cache list
        st.session_state['annt_cache'] = st.session_state['annt_cache'][:-1]
        return prev_index
    else:
        return None

