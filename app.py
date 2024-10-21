""" Streamlit dashboard to interact with the data collected """

import json
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import supervision as sv
import pims

from trackers import (
    Keypoint, 
    Keypoints, 
    PlayerTracker, 
    PlayerKeypointsTracker,
    BallTracker, 
    KeypointsTracker,
    TrackingRunner
)
from analytics import DataAnalytics
from visualizations.padel_court import padel_court_2d
from estimate_velocity import BallVelocityEstimator, ImpactType
from utils.video import save_video
from config import *

COLLECT_DATA = True


@st.fragment
def velocity_estimator(video_info: sv.VideoInfo):
        
    frame_index = st.slider(
        "Frames", 
        0, 
        video_info.total_frames, 
        1, 
    )

    image = np.array(st.session_state["video"][frame_index])
    st.image(image)

    with st.form("choose-frames"):
        frame_index_t0 = st.number_input(
            "First frame: ", 
            min_value=0,
            max_value=video_info.total_frames,
        )
        frame_index_t1 = st.number_input(
            "Second frame: ", 
            min_value=1,
            max_value=video_info.total_frames,
        )
        impact_type_ch = st.radio(
            "Impact type: ",
            options=["Floor", "Player"],
        )
        get_Vz = st.radio(
            "Consider difference in ball altitude: ",
            options=[False, True]
        )

        estimate = st.form_submit_button("Calculate velocity")

    if estimate:

        assert frame_index_t0 < frame_index_t1

        if st.session_state["players_tracker"] is None:
            st.error("Data missing.")
        else:
            estimator = BallVelocityEstimator(
                source_video_fps=video_info.fps,
                players_detections=st.session_state["players_tracker"].results.predictions,
                ball_detections=st.session_state["ball_tracker"].results.predictions,
                keypoints_detections=st.session_state["keypoints_tracker"].results.predictions,
            )

            if impact_type_ch == "Floor":
                impact_type = ImpactType.FLOOR
            elif impact_type_ch == "Player":
                impact_type = ImpactType.RACKET

            ball_velocity_data, ball_velocity = estimator.estimate_velocity(
                frame_index_t0, frame_index_t1, impact_type, get_Vz=get_Vz,
            )
            st.write(ball_velocity)
            st.write("Velocity: ", ball_velocity.norm)
            st.image(ball_velocity_data.draw_velocity(st.session_state["video"]))
            padel_court = padel_court_2d()
            padel_court.add_trace(
                go.Scatter(
                    x=[
                        ball_velocity_data.position_t0_proj[0],
                        ball_velocity_data.position_t1_proj[0],
                    ],
                    y=[
                        ball_velocity_data.position_t0_proj[1]*-1,
                        ball_velocity_data.position_t1_proj[1]*-1,
                    ],
                    marker= dict(
                        size=10,
                        symbol= "arrow-bar-up", 
                        angleref="previous",
                    ),
                )                    
            )
            st.plotly_chart(padel_court)


if "video" not in st.session_state:
    st.session_state["video"] = None

if "df" not in st.session_state:
    st.session_state["df"] = None

if "fixed_keypoints_detection" not in st.session_state:
    st.session_state["fixed_keypoints_detection"] = None

if "players_keypoints_tracker" not in st.session_state:
    st.session_state["players_keypoints_tracker"] = None

if "players_tracker" not in st.session_state:
    st.session_state["players_tracker"] = None

if "ball_tracker" not in st.session_state:
    st.session_state["ball_tracker"] = None

if "keypoints_tracker" not in st.session_state:
    st.session_state["keypoints_tracker"] = None

if "runner" not in st.session_state:
    st.session_state["runner"] = None

st.title("Padel Analytics")

with st.form("run-video"):
    upload_video_path = st.text_input(
        "Upload video: ",
        INPUT_VIDEO_PATH,
    )
    upload_video = st.form_submit_button("Upload")

if upload_video or st.session_state["video"] is not None:

    if upload_video:
        st.session_state["df"] = None
        os.system(f"ffmpeg -y -i {upload_video_path} -vcodec libx264 tmp.mp4")
    
    if st.session_state["df"] is None:

        with st.spinner("Analysing video ..."):
    
            video_info = sv.VideoInfo.from_video_path(video_path="tmp.mp4")  
            fps, w, h, total_frames = (
                video_info.fps, 
                video_info.width,
                video_info.height,
                video_info.total_frames,
            ) 
            
            if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None:
                with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:
                    SELECTED_KEYPOINTS = json.load(f)

            st.session_state["fixed_keypoints_detection"] = Keypoints(
                [
                    Keypoint(
                        id=i,
                        xy=tuple(float(x) for x in v)
                    )
                    for i, v in enumerate(SELECTED_KEYPOINTS)
                ]
            )

            keypoints_array = np.array(SELECTED_KEYPOINTS)
            # Polygon to filter person detections inside padel court
            polygon_zone = sv.PolygonZone(
                np.concatenate(
                    (
                        np.expand_dims(keypoints_array[0], axis=0), 
                        np.expand_dims(keypoints_array[1], axis=0), 
                        np.expand_dims(keypoints_array[-1], axis=0), 
                        np.expand_dims(keypoints_array[-2], axis=0),
                    ),
                    axis=0
                ),
                frame_resolution_wh=video_info.resolution_wh,
            )

            # Instantiate trackers
            st.session_state["players_tracker"] = PlayerTracker(
                PLAYERS_TRACKER_MODEL,
                polygon_zone,
                batch_size=PLAYERS_TRACKER_BATCH_SIZE,
                annotator=PLAYERS_TRACKER_ANNOTATOR,
                show_confidence=True,
                load_path=PLAYERS_TRACKER_LOAD_PATH,
                save_path=PLAYERS_TRACKER_SAVE_PATH,
            )

            st.session_state["player_keypoints_tracker"] = PlayerKeypointsTracker(
                PLAYERS_KEYPOINTS_TRACKER_MODEL,
                train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
                batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
                load_path=PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
                save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
            )

            st.session_state["ball_tracker"] = BallTracker(
                BALL_TRACKER_MODEL,
                BALL_TRACKER_INPAINT_MODEL,
                batch_size=BALL_TRACKER_BATCH_SIZE,
                median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
                median=None,
                load_path=BALL_TRACKER_LOAD_PATH,
                save_path=BALL_TRACKER_SAVE_PATH,
            )

            st.session_state["keypoints_tracker"] = KeypointsTracker(
                model_path=KEYPOINTS_TRACKER_MODEL,
                batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
                model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
                fixed_keypoints_detection=st.session_state["fixed_keypoints_detection"],
                load_path=KEYPOINTS_TRACKER_LOAD_PATH,
                save_path=KEYPOINTS_TRACKER_SAVE_PATH,
            )

            runner = TrackingRunner(
                trackers=[
                    st.session_state["players_tracker"], 
                    st.session_state["player_keypoints_tracker"], 
                    st.session_state["ball_tracker"],
                    st.session_state["keypoints_tracker"],    
                ],
                video_path="tmp.mp4",
                inference_path=OUTPUT_VIDEO_PATH,
                start=0,
                end=MAX_FRAMES,
                collect_data=COLLECT_DATA,
            )

            runner.run()

            st.session_state["runner"] = runner

            st.session_state["df"]  = runner.data_analytics.into_dataframe(
                runner.video_info.fps,
            )

            st.success("Done.")
    
    st.session_state["video"] = pims.Video("tmp.mp4")
    st.subheader("Uploaded Video")
    st.video("tmp.mp4")
    
    estimate_velocity = st.checkbox("Calculate Ball Velocity")
    if estimate_velocity:
        st.write("Select a frame to calculate ball velocity:")
        velocity_estimator(st.session_state["runner"].video_info)
    
    if st.session_state["df"] is not None:
        st.header("Collected data")
        st.write("First 5 rows")
        st.dataframe(st.session_state["df"].head())
        st.markdown(f"- Number of rows: {len(st.session_state["df"])}")
        # st.write("- Columns: ")
        # st.write(st.session_state["df"].columns)

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

        players_data = {
            "player_id": [],
            "total_distance_m": [],
            "mean_velocity_km/h": [],
            "maximum_velocity_km/h": [],
        }
        for player_id in (1, 2, 3, 4):
            players_data["player_id"].append(player_id)
            players_data["total_distance_m"].append(
                st.session_state["df"][
                    f"player{player_id}_distance"
                ].sum()
            )
            players_data["mean_velocity_km/h"].append(
                st.session_state["df"][
                    f"player{player_id}_V{velocity_type}4"
                ].abs().mean() * 3.6,
            )
            players_data["maximum_velocity_km/h"].append(
                st.session_state["df"][
                    f"player{player_id}_V{velocity_type}4"
                ].abs().max() * 3.6,
            )

        st.dataframe(pd.DataFrame(players_data).set_index("player_id"))

        st.subheader("Players velocity as a function of time")

        st.plotly_chart(fig)

        st.subheader("Analyze players position, velocity and acceleration")
        
        col1, col2 = st.columns((1, 1))

        with col1:
            player_choice = st.radio("Player: ", options=[1, 2, 3, 4])
        
        with col2:
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
                ].abs() * 3.6,
                marker=dict(
                    color=df_scatter[
                            f"player{player_choice}_V{velocity_type}4"
                    ].abs() * 3.6,
                    size=12,
                    showscale=True,
                    colorscale="jet",
                    cmin=min_value * 3.6,
                    cmax=max_value * 3.6,
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
            "time <= @time_span"
        )
        padel_court.add_trace(
            go.Scatter(
                x=df_time[f"player{player_choice}_x"],
                y=df_time[f"player{player_choice}_y"] * -1,
                mode="markers",
                name=f"Player {player_choice}",
                text=df_time[
                    f"player{player_choice}_V{velocity_type}4"
                ].abs() * 3.6,
                marker=dict(
                    color=df_time[
                            f"player{player_choice}_V{velocity_type}4"
                    ].abs() * 3.6,
                    size=12,
                    showscale=True,
                    colorscale="jet",
                    cmin=min_value * 3.6,
                    cmax=max_value * 3.6,
                )
            )
        )
        st.plotly_chart(padel_court)

        

        def plotly_fig2array(fig):
            """
            Convert a plotly figure to numpy array
            """
            import io
            from PIL import Image
            print("HERE3")
            fig_bytes = fig.to_image(format="png")
            print("HERE4")
            buf = io.BytesIO(fig_bytes)
            img = Image.open(buf)
            return np.asarray(img)

        def court_frames(player_choice, velocity_type):

            padel_court = padel_court_2d()

            for t in st.session_state["df"]["time"]:

                print("HERE1")
    
                x_values = st.session_state["df"].query(
                    "time <= @t"
                )[f"player{player_choice}_x"]

                y_values = st.session_state["df"].query(
                    "time <= @t"
                )[f"player{player_choice}_y"] * -1

                v_values = st.session_state["df"].query(
                    "time <= @t"
                )[f"player{player_choice}_V{velocity_type}4"].abs() * 3.6

                padel_court.add_trace(
                    go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode="markers",
                                name=f"Player {player_choice}",
                                text=v_values,
                                marker=dict(
                                    color=v_values,
                                    size=12,
                                    showscale=True,
                                    colorscale="jet",
                                    cmin=min_value * 3.6,
                                    cmax=max_value * 3.6,
                                )
                            )
                )

                print("HERE2")

                yield plotly_fig2array(padel_court)

        # for frame in court_frames(player_choice, velocity_type):
        #     print(type(frame))
        #    break    

        # save_video(
        #     court_frames(player_choice, velocity_type), 
        #   "positions.mp4", 
        #     fps=st.session_state["runner"].video_info.fps,
        #    w=st.session_state["runner"].video_info.width,
        #    h=st.session_state["runner"].video_info.height,
        #)

        

        
        
        

        
        
        

 
        

