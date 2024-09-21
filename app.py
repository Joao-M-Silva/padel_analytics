import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from analytics import DataAnalytics
from visualizations.padel_court import padel_court_2d


FPS = 25

if "df" not in st.session_state:
    st.session_state["df"] = None

with st.sidebar:
    with st.form("data-form"):

        path = st.text_input("Data path")
        FPS = st.number_input("Video FPS", 25)
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        with open(path) as f:
            data = json.load(f)
        
        data_analytics = DataAnalytics.from_dict(data)
        df = data_analytics.into_dataframe(FPS)
        st.session_state["df"] = df

st.title("Padel Analytics")

if st.session_state["df"] is not None:
    st.header("Loaded dataframe")
    st.write("First 20 lines")
    st.dataframe(st.session_state["df"].head(20))
    st.markdown(f"- Number of lines: {len(st.session_state["df"])}")
    st.write("- Columns: ")
    st.write(st.session_state["df"].columns)

    st.write("Player velocity as a function of time")
    velocity_type_choice = st.radio(
        "Type", 
        ["Horizontal", "Vertical", "Absolute"],
    )
    velocity_type_mapper = {
        "Horizontal": "x",
        "Vertical": "y",
        "Absolute": "norm",
    }
    velocity_type = velocity_type_mapper[velocity_type_choice]
    fig = go.Figure()
    padel_court = padel_court_2d()
    for player_id in (1, 2, 3, 4):
        fig.add_trace(
            go.Scatter(
                x=st.session_state["df"]["time"], 
                y=np.abs(
                    st.session_state["df"][
                        f"player{player_id}_V{velocity_type}4"
                    ].to_numpy()
                ),
                mode='lines',
                name=f'Player {player_id}',
            ),
        )
        st.write(f"Player {player_id}")
        st.write(
            "- Total distance (m): ",
            st.session_state["df"][
                f"player{player_id}_distance"
            ].sum()
        )
        st.write(
            " - Mean Velocity (m/s): ", 
            st.session_state["df"][
                f"player{player_id}_V{velocity_type}4"
            ].abs().mean(),
        )
        st.write(
            "- Maximum Velocity (m/s): ", 
            st.session_state["df"][
                f"player{player_id}_V{velocity_type}4"
            ].abs().max(),
        )

    st.plotly_chart(fig)



    
    
    player_choice = st.radio("Player: ", options=[1, 2, 3, 4])
    min_value = st.session_state["df"][
        f"player{player_choice}_V{velocity_type}4"
    ].abs().min()
    max_value = st.session_state["df"][
        f"player{player_choice}_V{velocity_type}4"
    ].abs().max()
    velocity_interval = st.slider(
        "Velocity Interval",
        min_value, 
        max_value,
        (min_value, max_value),
    )
    st.session_state["df"]["QUERY_VELOCITY"] = st.session_state["df"][
        f"player{player_choice}_V{velocity_type}4"
    ].abs()
    min_choice = velocity_interval[0]
    max_choice = velocity_interval[1]
    df_scatter = st.session_state["df"].query(
        "@min_choice <= QUERY_VELOCITY <= @max_choice"
    )
        
    padel_court.add_trace(
        go.Scatter(
            x=df_scatter[f"player{player_choice}_x"],
            y=df_scatter[f"player{player_choice}_y"] * -1,
            mode="markers",
            name=f"Player {player_choice}",
            text=df_scatter[
                f"player{player_choice}_V{velocity_type}4"
            ].abs(),
            marker=dict(
                color=df_scatter[
                        f"player{player_choice}_V{velocity_type}4"
                ].abs(),
                size=12,
                showscale=True,
                colorscale="jet",
            )
        )
    )

    st.plotly_chart(padel_court)

    padel_court = padel_court_2d()
    time_span = st.slider(
        "Time Interval",
        0.0, 
        st.session_state["df"]["time"].max(),
    )
    df_time = st.session_state["df"].query(
        "time >= @time_span"
    )
    padel_court.add_trace(
        go.Scatter(
            x=df_time[f"player{player_choice}_x"],
            y=df_time[f"player{player_choice}_y"] * -1,
            mode="markers",
            name=f"Player {player_choice}",
            text=df_time[
                f"player{player_choice}_V{velocity_type}4"
            ].abs(),
            marker=dict(
                color=df_time[
                        f"player{player_choice}_V{velocity_type}4"
                ].abs(),
                size=12,
                showscale=True,
                colorscale="jet",
            )
        )
    )
    st.plotly_chart(padel_court)


