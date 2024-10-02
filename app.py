import json
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
import supervision as sv
import pims

from trackers import Keypoint, PlayerTracker, BallTracker, KeypointsTracker
from analytics import DataAnalytics
from visualizations.padel_court import padel_court_2d
from estimate_velocity import BallVelocityEstimator, ImpactType
from trackers.velocity_in_time import get_velocity_vector_per_frame_interval

if "video" not in st.session_state:
    st.session_state["video"] = None

if "df" not in st.session_state:
    st.session_state["df"] = None

if "fixed_keypoints_detection" not in st.session_state:
    st.session_state["fixed_keypoints_detection"] = None

if "keypoints_detections" not in st.session_state:
    st.session_state["keypoints_detections"] = None

if "players_detections" not in st.session_state:
    st.session_state["players_detections"] = None

if "ball_detections" not in st.session_state:
    st.session_state["ball_detections"] = None

# Models weights
PLAYERS_TRACKER_MODEL = "yolov8m.pt"
BALL_TRACKER_MODEL = "./weights/ball_detection/TrackNet_best.pt"
BALL_TRACKER_INPAINT_MODEL = "./weights/ball_detection/InpaintNet_best.pt"
KEYPOINTS_TRACKER_MODEL = "./runs/keypoints/train2/weights/best.pt"
PLAYERS_KEYPOINTS_TRACKER_MODEL = "./runs/pose/train3/weights/best.pt"

st.title("Padel Analytics")

with st.form("upload-video"):
    UPLOAD_VIDEO_PATH = st.text_input(
        "Upload analysed video: ",
        "./test_all_detections.mp4",
    )
    upload_video = st.form_submit_button("Upload")

if upload_video or st.session_state["video"] is not None:

    if upload_video:
        os.system(f"ffmpeg -y -i {UPLOAD_VIDEO_PATH} -vcodec libx264 tmp.mp4")
    
    video_info = sv.VideoInfo.from_video_path(video_path="tmp.mp4")
    fps, w, h, total_frames = (
        video_info.fps, 
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )
    frame_generator = sv.get_video_frames_generator(
        "tmp.mp4",
        start=0,
        stride=1,
        end=300,
    )

    st.video("tmp.mp4")
    st.session_state["video"] = pims.Video("tmp.mp4")

    @st.fragment
    def velocity_estimator(fps: int):
        
        frame_index = st.slider(
            "Frames", 
            0, 
            total_frames, 
            1, 
        )

        image = np.array(st.session_state["video"][frame_index])
        st.image(image)

        with st.form("choose-frames"):
            frame_index_t0 = st.number_input(
                "First frame: ", 
                min_value=0,
                max_value=total_frames,
            )
            frame_index_t1 = st.number_input(
                "Second frame: ", 
                min_value=1,
                max_value=total_frames,
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

            if st.session_state["players_detections"] is None:
                st.error("Data missing.")
            else:
                estimator = BallVelocityEstimator(
                    source_video_fps=fps,
                    players_detections=st.session_state["players_detections"],
                    ball_detections=st.session_state["ball_detections"],
                    keypoints_detections=st.session_state["keypoints_detections"],
                )

                if impact_type_ch == "Floor":
                    impact_type = ImpactType.FLOOR
                elif impact_type_ch == "Player":
                    impact_type = ImpactType.RACKET

                ball_velocity_data, ball_velocity = estimator.estimate_velocity(
                    frame_index_t0, frame_index_t1, impact_type, get_Vz=get_Vz,
                )
                st.write(ball_velocity)
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
                

    estimate_velocity = st.checkbox("Calculate Ball Velocity")
    if estimate_velocity:
        st.write("Select a frame to calculate ball velocity:")
        velocity_estimator(fps)

    with st.sidebar:
        with st.form("data-form"):

            st.subheader("Upload trackers results")
            PLAYERS_DETECTIONS_LOAD_PATH = st.text_input(
                "Players detections: ",
                "cache/player_detections.json"
            )
            BALL_DETECTIONS_LOAD_PATH = st.text_input(
                "Ball detections: ",
                "cache/ball_detections.json",
            )
            FIXED_KEYPOINTS_DETECTIONS_LOAD_PATH = st.text_input(
                "Court keypoints detections: ",
                "cache/fixed_keypoints_detection.json",
            )
            KEYPOINTS_DETECTIONS_LOAD_PATH = st.text_input(
                "Court keypoints detections: ",
                "cache/keypoints_detections.json",
            )
            PLAYERS_KEYPOINTS_DETECTIONS_LOAD_PATH = st.text_input(
                "Players keypoints detections: ",
                "cache/players_keypoints_detections.json",
            )

            st.subheader("Analysis data: ")
            data_path = st.text_input("Data path", "data.json")
            
            results_submitted = st.form_submit_button("Submit")
    
    if results_submitted:
        with open(FIXED_KEYPOINTS_DETECTIONS_LOAD_PATH) as f:
            fixed_keypoints_detection = json.load(f)

        keypoints_array = np.array(fixed_keypoints_detection)
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

        st.session_state["fixed_keypoints_detection"] = [
            Keypoint(
                id=i,
                xy=tuple(float(x) for x in v)
            )
            for i, v in enumerate(fixed_keypoints_detection)
        ]
            
        player_tracker = PlayerTracker(
            PLAYERS_TRACKER_MODEL, 
            video_info,
            polygon_zone,
        )
        st.session_state["players_detections"] = player_tracker.load_detections(
            PLAYERS_DETECTIONS_LOAD_PATH,
        )
        assert len(st.session_state["players_detections"]) == total_frames

        ball_tracker = BallTracker(
            BALL_TRACKER_MODEL,
            BALL_TRACKER_INPAINT_MODEL,
        )
        st.session_state["ball_detections"] = ball_tracker.load_detections(BALL_DETECTIONS_LOAD_PATH)
        assert len(st.session_state["ball_detections"]) == total_frames
        
        keypoints_tracker = KeypointsTracker(
            model_path=KEYPOINTS_TRACKER_MODEL,
            model_type="yolo",
            fixed_keypoints_detection=fixed_keypoints_detection,
        )
        st.session_state["keypoints_detections"] = keypoints_tracker.load_detections(KEYPOINTS_DETECTIONS_LOAD_PATH)
        assert len(st.session_state["keypoints_detections"]) == total_frames
            
        with open(data_path) as f:
            data = json.load(f)
            
        data_analytics = DataAnalytics.from_dict(data)
        df = data_analytics.into_dataframe(fps)
        st.session_state["df"] = df

# st.session_state["df"] =  None
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


