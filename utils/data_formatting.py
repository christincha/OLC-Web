import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict
from typing import Set
from streamlit_plotly_events import plotly_events
import plotly.express as px
import plotly.graph_objs as go

def query_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds x y location information to the data frame
    Inputs:
        df: 2D projection of the latent representation in form of pandas data array
    """
    if st.session_state['num_cm_d'] == '2d':
        df["x-y"] = (
            (100 * df["x"]).astype(int).astype(str)
            + "-"
            + (100 * df["y"]).astype(int).astype(str)
        )
    else:
        df["x-y"] = (
            (100 * df["x"]).astype(int).astype(str)
            + "-"
            + (100 * df["y"]).astype(int).astype(str)
            + '-'
            ++ (100 * df["z"]).astype(int).astype(str)
        )
    # Sets default value informing selection state
    df["selected"] = True
    for q in ["x-y"]:
        print(st.session_state[f"{q}_query"])
        if st.session_state[f"{q}_query"]:
            df.loc[~df[q].isin(st.session_state[f"{q}_query"]), "selected"] = False

    return df

def update_state(current_query: Dict[str, Set], df):
    """
    Updates the sample selection states
    Inputs:
        current_query: dictionary of samples selected for displaying
        df: pandas data from including projected latent representation
    """
    rerun = False
    for q in ["x-y"]:
        if current_query[f"{q}_query"] - st.session_state[f"{q}_query"]:
            # Updates the state when the state is different
            st.session_state[f"{q}_query"] = current_query[f"{q}_query"]
            rerun = True
            cur_index = np.where(df["x-y"] == list(current_query[f"{q}_query"])[0])
            if len(cur_index) > 0:
                cur_index = np.where(df["x-y"] == list(current_query[f"{q}_query"])[0])[0][0]
                st.session_state[f"x-y-sel"] = [cur_index]
            print('selection', list(current_query[f"{q}_query"]),
                  'cor id', np.where(df["x-y"] == list(current_query[f"{q}_query"])[0]))
    return rerun

def reset_state_callback():
    """Resets all filters and increments counter in Streamlit Session State
    """
    st.session_state.counter = 1 + st.session_state.counter

    for q in ["x-y"]:
        st.session_state[f"{q}_query"] = set()


def build_figure(df: pd.DataFrame) -> go.Figure:
    """
    Builds catter plot figure
    Input:
        df: projected latent representation in pandas data format
    """
    if st.session_state['num_cm_d'] == "2d":
        fig = px.scatter(
            df,
            "x",
            "y",
            color="selected",
            color_discrete_sequence=["gray", "red"],
            category_orders={"selected": [False, True]},
            hover_data=[
                "x",
                "y",
            ],
            width=400, height=400
        )
    elif st.session_state['num_cm_d'] == "3d":
        fig = px.scatter_3d(
            df,
            "x",
            "y",
            "z",
            color="selected",
            color_discrete_sequence=["gray", "red"],
            category_orders={"selected": [False, True]},
            hover_data=[
                "x",
                "y",
                "z"
            ],
            width=400, height=400
        )

    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig['layout'].update(margin=dict(l=0, r=0, b=0, t=0))
    return fig

# def render_preview_ui(df: pd.DataFrame):
#     """
#     Renders an expander with content of DataFrame and Streamlit Session State
#     """
#     with st.expander("Preview"):
#         l, r = st.columns(2)
#         l.dataframe(
#             df,
#         )
#         r.json(
#             {
#                 k: v
#                 for k, v in st.session_state.to_dict().items()
#                 if f'_{st.session_state["counter"]}' not in k
#             }
#         )


def render_plotly_ui(transformed_df: pd.DataFrame) -> tuple:
    """
    Generates streamlit plot.
    Inputs:
        transformed_df: reduced latent representation in pandas data format
    """

    cluster_map_figure = build_figure(transformed_df)

    # Enables streamlit click event
    sample_selected = plotly_events(
        cluster_map_figure,
        click_event=True, hover_event=False,
        key=f"x-y_{st.session_state.counter}",
    )

    # Stores selected samples in click events
    current_query = {}
    if st.session_state['num_cm_d'] == "2d":
        current_query["x-y_query"] = {
            f"{int(100*el['x'])}-{int(100*el['y'])}" for el in sample_selected[:1]
        }
    else:
        print(sample_selected)
        current_query["x-y_query"] = {
            f"{int(100*el['x'])}-{int(100*el['y'])}-{int(100*el['z'])}" for el in sample_selected[:1]
        }

    return current_query, sample_selected