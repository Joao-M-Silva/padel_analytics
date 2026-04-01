""" Streamlit dashboard to interact with the data collected """

import datetime
import hashlib
import json
import numpy as np
import os
import re
import subprocess
import tempfile
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import supervision as sv
import pims

# Ensure required directories exist at startup
from config import RESULTS_DIR
for _d in (
    "./cache",
    "./weights/players_detection",
    "./weights/ball_detection",
    "./weights/players_keypoints_detection",
    "./weights/court_keypoints_detection",
    RESULTS_DIR,
):
    os.makedirs(_d, exist_ok=True)

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
from analytics.data_analytics import zone_breakdown, partner_synchrony, coaching_kpis
from analytics.hit_detection import detect_hits
from analytics.shot_classifier import classify_hits
from analytics.rally_analysis import segment_rallies, enrich_rallies, analyse_rallies_with_claude
from visualizations.padel_court import (
    padel_court_2d, padel_court_2d_heatmap, padel_court_2d_zones,
    padel_court_heatmap_kde,
)
from estimate_velocity import BallVelocityEstimator, ImpactType
from utils.video import save_video
from config import *

COLLECT_DATA = True

SPEED_ZONES = [
    ("Standing", 0, 2),
    ("Walking", 2, 7),
    ("Jogging", 7, 15),
    ("Running", 15, 20),
    ("Sprinting", 20, 9999),
]


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
                    marker=dict(
                        size=10,
                        symbol="arrow-bar-up",
                        angleref="previous",
                    ),
                )
            )
            st.plotly_chart(padel_court)


# --- Session state initialisation ---
for _key in ("video", "df", "fixed_keypoints_detection",
             "players_keypoints_tracker", "players_tracker",
             "ball_tracker", "keypoints_tracker", "runner", "analytics",
             "my_player", "player_names", "current_video_id"):
    if _key not in st.session_state:
        st.session_state[_key] = None

# ── Load persisted player profile ────────────────────────────────────────────
_profile_path = "./cache/player_profile.json"
if st.session_state["my_player"] is None and os.path.exists(_profile_path):
    try:
        _prof = json.load(open(_profile_path))
        st.session_state["my_player"] = _prof.get("my_player", 1)
        st.session_state["player_names"] = _prof.get("player_names",
            {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"})
    except Exception:
        pass
if st.session_state["my_player"] is None:
    st.session_state["my_player"] = 1
if st.session_state["player_names"] is None:
    st.session_state["player_names"] = {
        "1": "Player 1", "2": "Player 2",
        "3": "Player 3", "4": "Player 4",
    }

# ─────────────────────────────────────────────────────────────────────────────
st.title("Padel Analytics")

# ── Player identity sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("My Profile")
    _pnames = st.session_state["player_names"]
    _new_my_player = st.selectbox(
        "I am:",
        options=[1, 2, 3, 4],
        index=[1, 2, 3, 4].index(st.session_state["my_player"]),
        format_func=lambda p: f"Player {p} — {_pnames.get(str(p), f'Player {p}')}",
    )
    if _new_my_player != st.session_state["my_player"]:
        st.session_state["my_player"] = _new_my_player

    with st.expander("Player names"):
        for _pid in (1, 2, 3, 4):
            _pnames[str(_pid)] = st.text_input(
                f"Player {_pid}", value=_pnames.get(str(_pid), f"Player {_pid}"),
                key=f"pname_{_pid}",
            )
        if st.button("Save names"):
            st.session_state["player_names"] = _pnames
            _profile_data = {
                "my_player": st.session_state["my_player"],
                "player_names": _pnames,
            }
            with open(_profile_path, "w") as _pf:
                json.dump(_profile_data, _pf)
            st.success("Saved")

_pnames = st.session_state["player_names"]
_me = st.session_state["my_player"]

# ── Video history ─────────────────────────────────────────────────────────────
_index_path = f"{RESULTS_DIR}/index.json"
_history = json.load(open(_index_path)) if os.path.exists(_index_path) else []

if _history:
    def _history_label(e):
        _meta_path = f"{RESULTS_DIR}/{e['video_id']}_metadata.json"
        if os.path.exists(_meta_path):
            try:
                _m = json.load(open(_meta_path))
                _date = _m.get("date", e["processed_at"][:10])
                _result = _m.get("result", "")
                _club = _m.get("club", "")
                _parts = [_date]
                if _result:
                    _parts.append(_result)
                if _club:
                    _parts.append(_club)
                return " — ".join(_parts)
            except Exception:
                pass
        return f"{e['processed_at'][:10]} — {e['url'][:60]}"

    _options = {_history_label(e): e for e in reversed(_history)}
    _selected = st.selectbox(
        "Load a previously analysed video:",
        ["— analyse a new video —"] + list(_options.keys()),
    )
    if _selected != "— analyse a new video —":
        _entry = _options[_selected]
        _csv_path = f"{RESULTS_DIR}/{_entry['video_id']}.csv"
        _analytics_path = f"{RESULTS_DIR}/{_entry['video_id']}_analytics.json"
        if os.path.exists(_csv_path):
            st.session_state["df"] = pd.read_csv(_csv_path)
            st.session_state["current_video_id"] = _entry["video_id"]
            if os.path.exists(_analytics_path):
                with open(_analytics_path) as _af:
                    st.session_state["analytics"] = json.load(_af)
            else:
                st.session_state["analytics"] = None
            st.success(f"Loaded: {_selected}")

# --- In-progress analysis indicator ---
_url_cache_path = "./cache/current_video_url.txt"
if os.path.exists(_url_cache_path):
    _in_progress_url = open(_url_cache_path).read().strip()
    _completed_urls = {e["url"] for e in _history}
    if _in_progress_url and _in_progress_url not in _completed_urls:
        st.warning(
            "**Analysis in progress or interrupted**\n\n"
            f"`{_in_progress_url[:100]}`\n\n"
            "Paste the URL below and click **Load Video** to resume — "
            "the download and any completed trackers will be skipped automatically."
        )

st.subheader("Load Video")
video_url = st.text_input(
    "Paste a Google Drive file link or direct video URL:",
    placeholder="https://drive.google.com/file/d/FILE_ID/view  or  https://example.com/video.mp4",
)
load_video = st.button("Load Video")

if load_video or st.session_state["video"] is not None:

    if load_video:
        if not video_url:
            st.error("Please enter a video URL.")
            st.stop()

        st.session_state["df"] = None

        _url_cache = "./cache/current_video_url.txt"
        _skip_download = (
            os.path.exists("tmp.mp4")
            and os.path.getsize("tmp.mp4") > 0
            and os.path.exists(_url_cache)
            and open(_url_cache).read().strip() == video_url.strip()
        )

        if _skip_download:
            st.info("Video already downloaded — resuming from cache.")
        else:
            with st.spinner("Downloading video..."):
                tmp_input_path = tempfile.mktemp(suffix=".mp4")

                gdrive_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", video_url)
                if gdrive_match or "drive.google.com" in video_url:
                    import gdown
                    file_id = gdrive_match.group(1) if gdrive_match else video_url
                    gdown.download(id=file_id, output=tmp_input_path, quiet=False, fuzzy=True)
                else:
                    response = requests.get(video_url, stream=True, timeout=300)
                    response.raise_for_status()
                    with open(tmp_input_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_input_path, "-vcodec", "libx264", "tmp.mp4"],
                check=True,
            )
            os.unlink(tmp_input_path)
            with open(_url_cache, "w") as _f:
                _f.write(video_url.strip())

    if st.session_state["df"] is None:

        with st.status("Analysing video...", expanded=True) as _status:

            def progress_callback(step_name, current, total):
                if step_name == "done":
                    return
                label = step_name.replace("_", " ").title()
                _status.write(f"▶ {label}  (step {current + 1} / {total})")

            _status.write("Setting up trackers...")
            video_info = sv.VideoInfo.from_video_path(video_path="tmp.mp4")
            fps, w, h, total_frames = (
                video_info.fps,
                int(video_info.width),
                int(video_info.height),
                video_info.total_frames,
            )

            if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None and os.path.exists(FIXED_COURT_KEYPOINTS_LOAD_PATH):
                with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:
                    SELECTED_KEYPOINTS = json.load(f)
                fixed_keypoints = Keypoints(
                    [
                        Keypoint(
                            id=i,
                            xy=tuple(float(x) for x in v)
                        )
                        for i, v in enumerate(SELECTED_KEYPOINTS)
                    ]
                )
                keypoints_array = np.array(SELECTED_KEYPOINTS, dtype=np.int32)
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
                    frame_resolution_wh=(w, h),
                )
            else:
                # No fixed keypoints file — detect keypoints per-frame; use full-frame polygon
                fixed_keypoints = None
                polygon_zone = sv.PolygonZone(
                    np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32),
                    frame_resolution_wh=(w, h),
                )

            st.session_state["fixed_keypoints_detection"] = fixed_keypoints

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
                conf=KEYPOINTS_TRACKER_CONF,
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

            runner.run(progress_callback=progress_callback)

            st.session_state["runner"] = runner

            df = runner.data_analytics.into_dataframe(runner.video_info.fps)
            st.session_state["df"] = df

            # Persist results for future sessions
            video_id = hashlib.md5(video_url.encode()).hexdigest()[:12]
            df.to_csv(f"{RESULTS_DIR}/{video_id}.csv", index=False)

            # Pre-compute and cache analytics
            _analytics_cache: dict = {
                "zone_breakdown": {
                    str(pid): zone_breakdown(df, pid)
                    for pid in (1, 2, 3, 4)
                },
                "partner_synchrony": {
                    "1_2": {
                        k: v for k, v in partner_synchrony(df, 1, 2).items()
                        if k in ("vertical_sync", "horizontal_sync", "avg_formation_width")
                    },
                    "3_4": {
                        k: v for k, v in partner_synchrony(df, 3, 4).items()
                        if k in ("vertical_sync", "horizontal_sync", "avg_formation_width")
                    },
                },
                "coaching_kpis": {
                    str(pid): coaching_kpis(df, pid) for pid in (1, 2, 3, 4)
                },
            }

            # Phase B: hit detection + rally segmentation (only if ball data present)
            if "ball_x" in df.columns and df["ball_x"].notna().any():
                _hits = detect_hits(df)
                _hits = classify_hits(_hits, df)
                _rallies = segment_rallies(df)
                _rallies = enrich_rallies(_rallies, df, _hits)
                _analytics_cache["hits"] = _hits
                _analytics_cache["rallies"] = _rallies

            _analytics_path = f"{RESULTS_DIR}/{video_id}_analytics.json"
            with open(_analytics_path, "w") as _af:
                json.dump(_analytics_cache, _af, indent=2)
            st.session_state["analytics"] = _analytics_cache
            st.session_state["current_video_id"] = video_id

            _history = json.load(open(_index_path)) if os.path.exists(_index_path) else []
            _history.append({
                "video_id": video_id,
                "url": video_url,
                "processed_at": datetime.datetime.now().isoformat(),
                "total_frames": runner.video_info.total_frames,
                "fps": runner.video_info.fps,
            })
            with open(_index_path, "w") as _f:
                json.dump(_history, _f, indent=2)

            _status.update(label="Analysis complete!", state="complete", expanded=False)

    st.session_state["video"] = pims.Video("tmp.mp4")

    with st.expander("Video & Tools", expanded=False):
        st.video("tmp.mp4")
        estimate_velocity = st.checkbox("Calculate Ball Velocity")
        if estimate_velocity:
            velocity_estimator(st.session_state["runner"].video_info)

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        _analytics = st.session_state.get("analytics") or {}
        _me = st.session_state["my_player"]
        _pnames = st.session_state["player_names"]
        _my_name = _pnames.get(str(_me), f"Player {_me}")
        _partner_id = {1: 2, 2: 1, 3: 4, 4: 3}[_me]
        _partner_name = _pnames.get(str(_partner_id), f"Player {_partner_id}")
        _has_ball_data = "ball_x" in df.columns and df["ball_x"].notna().any()

        # ── Match tagging form ────────────────────────────────────────────────
        _vid_id = st.session_state.get("current_video_id") or ""
        _meta_path = f"{RESULTS_DIR}/{_vid_id}_metadata.json" if _vid_id else None
        _existing_meta = {}
        if _meta_path and os.path.exists(_meta_path):
            try:
                _existing_meta = json.load(open(_meta_path))
            except Exception:
                pass

        with st.expander("Match Details (tag this session)", expanded=not bool(_existing_meta)):
            _m_col1, _m_col2 = st.columns(2)
            with _m_col1:
                _meta_date = st.date_input("Date", value=pd.to_datetime(
                    _existing_meta.get("date", datetime.date.today().isoformat())
                ).date())
                _meta_club = st.text_input("Club / Court", value=_existing_meta.get("club", ""))
            with _m_col2:
                _meta_result = st.text_input("Result (e.g. 6-3 6-2)", value=_existing_meta.get("result", ""))
                _meta_notes = st.text_input("Notes", value=_existing_meta.get("notes", ""))
            if st.button("Save match details") and _meta_path:
                _meta = {
                    "date": _meta_date.isoformat(),
                    "club": _meta_club,
                    "result": _meta_result,
                    "notes": _meta_notes,
                    "player_names": _pnames,
                }
                with open(_meta_path, "w") as _mf:
                    json.dump(_meta, _mf, indent=2)
                st.success("Match details saved.")

        # ── Match import from external tracker ───────────────────────────────
        with st.expander("Import from match tracker", expanded=False):
            st.caption(
                "Upload a JSON file exported from your match tracker. "
                "Expected format: {\"date\": ..., \"result\": ..., \"players\": {\"1\": ..., \"2\": ..., \"3\": ..., \"4\": ...}, \"club\": ...}"
            )
            _uploaded_meta = st.file_uploader("Match tracker export (.json)", type="json", key="meta_upload")
            if _uploaded_meta and _meta_path:
                try:
                    _imported = json.load(_uploaded_meta)
                    if "players" in _imported:
                        for _pk, _pv in _imported["players"].items():
                            _pnames[str(_pk)] = _pv
                        st.session_state["player_names"] = _pnames
                    with open(_meta_path, "w") as _mf:
                        json.dump({**_imported, "player_names": _pnames}, _mf, indent=2)
                    st.success("Imported match details. Player names updated.")
                    st.rerun()
                except Exception as _e:
                    st.error(f"Could not parse file: {_e}")

        # ── Export ─────────────────────────────────────────────────────────
        with st.expander("Export", expanded=False):
            _ex1, _ex2 = st.columns(2)
            with _ex1:
                st.download_button("Download CSV", data=df.to_csv(index=False).encode(),
                                   file_name="padel_analytics.csv", mime="text/csv")
            with _ex2:
                if os.path.exists(OUTPUT_VIDEO_PATH):
                    with open(OUTPUT_VIDEO_PATH, "rb") as _vid_f:
                        st.download_button("Download Annotated Video", data=_vid_f.read(),
                                           file_name="results.mp4", mime="video/mp4")

        # ═══════════════════════════════════════════════════════════════════
        # TAB NAVIGATION
        # ═══════════════════════════════════════════════════════════════════
        _tab_summary, _tab_position, _tab_movement, _tab_shots, _tab_partner, _tab_ai, _tab_overview = st.tabs([
            f"My Summary",
            "My Positioning",
            "My Movement",
            "My Shots",
            "My Partner",
            "AI Coaching",
            "Match Overview",
        ])

        # ── helpers ──────────────────────────────────────────────────────────
        def _sync_label(r):
            if r is None:
                return "N/A", ""
            if r >= 0.65:
                return f"{r:.2f}", "🟢 Good"
            if r >= 0.40:
                return f"{r:.2f}", "🟡 Developing"
            return f"{r:.2f}", "🔴 Needs work"

        # ═══════════════════════════════════════════════════════════════════
        # TAB: MY SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        with _tab_summary:
            st.subheader(f"{_my_name}'s Match Summary")

            dist = df[f"player{_me}_distance"].sum()
            max_v = df[f"player{_me}_Vnorm4"].abs().max() * 3.6
            avg_v = df[f"player{_me}_Vnorm4"].abs().mean() * 3.6
            _z = (_analytics.get("zone_breakdown") or {}).get(str(_me)) or zone_breakdown(df, _me)
            _kpi = (_analytics.get("coaching_kpis") or {}).get(str(_me)) or coaching_kpis(df, _me)

            _s1, _s2, _s3, _s4 = st.columns(4)
            with _s1:
                st.metric("Distance", f"{dist:.0f} m")
            with _s2:
                st.metric("Max speed", f"{max_v:.1f} km/h")
            with _s3:
                st.metric("Avg speed", f"{avg_v:.1f} km/h")
            with _s4:
                st.metric("Net approaches", _kpi["net_approach_count"])

            # Zone summary bar
            fig_zone_single = go.Figure()
            for _zk, _zc in [("front", "#22c55e"), ("transition", "#f59e0b"), ("back", "#ef4444")]:
                fig_zone_single.add_trace(go.Bar(
                    name=_zk.capitalize(), x=["Court zones"], y=[_z[_zk]],
                    marker_color=_zc,
                    text=[f"{_z[_zk]:.0f}%"], textposition="inside",
                ))
            fig_zone_single.update_layout(
                barmode="stack", height=90,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(orientation="h", y=2.0, x=0),
                yaxis=dict(showticklabels=False, showgrid=False),
                xaxis=dict(showticklabels=False, showgrid=False),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_zone_single, use_container_width=True)

            st.caption("Front: net zone (|y|<3m) · Transition: 3–6m · Back: >6m")

            # Secondary KPIs
            _k1, _k2, _k3 = st.columns(3)
            with _k1:
                st.metric("Time in no-man's land", f"{_kpi['time_in_nomansland_pct']:.1f}%",
                          help="% of frames in the transition zone (3–6m) — lower is better")
            with _k2:
                st.metric("Direction changes", _kpi["change_of_direction_count"],
                          help="Lateral direction reversals — higher indicates active lateral movement")
            with _k3:
                st.metric("Sprint bursts (>20 km/h)", _kpi["peak_sprint_count"])

        # ═══════════════════════════════════════════════════════════════════
        # TAB: MY POSITIONING
        # ═══════════════════════════════════════════════════════════════════
        with _tab_position:
            st.subheader(f"{_my_name}'s Court Positioning")

            _player_half = "top" if _me in (1, 2) else "bottom"
            _heat_fig = padel_court_heatmap_kde(
                df[f"player{_me}_x"],
                df[f"player{_me}_y"],
                player_half=_player_half,
                width=420,
                title=f"{_my_name} — position density",
            )
            _ph_col, _pz_col = st.columns([2, 1])
            with _ph_col:
                st.plotly_chart(_heat_fig, use_container_width=False)
            with _pz_col:
                st.subheader("Zone breakdown")
                _z = (_analytics.get("zone_breakdown") or {}).get(str(_me)) or zone_breakdown(df, _me)
                for _zname, _zcolor, _zdesc in [
                    ("front", "#22c55e", "Net zone — attacking position"),
                    ("transition", "#f59e0b", "No-man's land — should minimise"),
                    ("back", "#ef4444", "Back court — defensive position"),
                ]:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:10px;margin:8px 0'>"
                        f"<div style='width:14px;height:14px;border-radius:3px;background:{_zcolor}'></div>"
                        f"<div><b>{_z[_zname]:.1f}%</b> {_zname.capitalize()}<br>"
                        f"<span style='font-size:0.75rem;color:#888'>{_zdesc}</span></div></div>",
                        unsafe_allow_html=True,
                    )
                _kpi = (_analytics.get("coaching_kpis") or {}).get(str(_me)) or coaching_kpis(df, _me)
                st.metric("Net approaches", _kpi["net_approach_count"], help="Times crossed into |y|<2m")
                _lb = _kpi["lateral_bias"]
                _bias_label = "→ Right-biased" if _lb > 0.1 else ("← Left-biased" if _lb < -0.1 else "Balanced")
                st.metric("Lateral bias", _bias_label, help="Direction of predominant lateral movement")

        # ═══════════════════════════════════════════════════════════════════
        # TAB: MY MOVEMENT
        # ═══════════════════════════════════════════════════════════════════
        with _tab_movement:
            st.subheader(f"{_my_name}'s Movement Profile")

            # Speed zones donut
            _speeds = df[f"player{_me}_Vnorm4"].abs() * 3.6
            _n = max(len(_speeds), 1)
            _zone_pcts = [
                100 * ((_speeds >= lo) & (_speeds < hi)).sum() / _n
                for _, lo, hi in SPEED_ZONES
            ]
            _zone_names = [z[0] for z in SPEED_ZONES]
            _zone_colors_mov = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

            fig_speed = go.Figure(go.Pie(
                labels=_zone_names, values=_zone_pcts,
                marker_colors=_zone_colors_mov,
                hole=0.55,
                textinfo="label+percent",
            ))
            fig_speed.update_layout(height=320, showlegend=False,
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(l=20, r=20, t=20, b=20),
                                    font=dict(color="#d0d0e0"))
            st.plotly_chart(fig_speed, use_container_width=True)

            # Velocity over time
            fig_vel = go.Figure()
            fig_vel.add_trace(go.Scatter(
                x=df["time"],
                y=df[f"player{_me}_Vnorm4"].abs() * 3.6,
                mode="lines",
                line=dict(color="#6366f1", width=1),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.15)",
                name=_my_name,
            ))
            fig_vel.update_layout(
                yaxis_title="Speed (km/h)", xaxis_title="Time (s)",
                height=220, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#2d3139"), xaxis=dict(gridcolor="#2d3139"),
            )
            st.plotly_chart(fig_vel, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════════
        # TAB: MY SHOTS
        # ═══════════════════════════════════════════════════════════════════
        with _tab_shots:
            st.subheader(f"{_my_name}'s Shots")

            if not _has_ball_data:
                st.info("Ball position data not yet available. Re-run the pipeline to enable shot analysis.")
            else:
                _hits = _analytics.get("hits")
                if _hits is None:
                    _hits = classify_hits(detect_hits(df), df)

                _my_hits = [h for h in (_hits or []) if h["player_id"] == _me]
                if not _my_hits:
                    st.write("No hit events detected for this player.")
                else:
                    st.metric("Total hits", len(_my_hits))

                    # Horizontal bar chart — one bar per shot type, sorted by count
                    from collections import Counter
                    _shot_counts = Counter(h.get("shot_type", "unknown") for h in _my_hits)
                    _sorted_shots = sorted(_shot_counts.items(), key=lambda x: x[1], reverse=True)
                    _shot_names = [s[0] for s in _sorted_shots]
                    _shot_vals = [s[1] for s in _sorted_shots]
                    _total_hits = len(_my_hits)

                    _shot_palette = {
                        "smash": "#6366f1", "bandeja": "#22c55e", "vibora": "#f59e0b",
                        "volley": "#14b8a6", "chiquita": "#f97316", "globo": "#8b5cf6",
                        "bajada": "#ec4899", "unknown": "#555",
                    }

                    fig_my_shots = go.Figure(go.Bar(
                        x=_shot_vals,
                        y=_shot_names,
                        orientation="h",
                        marker_color=[_shot_palette.get(s, "#888") for s in _shot_names],
                        text=[f"{v} ({100*v//_total_hits}%)" for v in _shot_vals],
                        textposition="outside",
                    ))
                    fig_my_shots.update_layout(
                        height=max(200, 45 * len(_shot_names)),
                        xaxis_title="Count",
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="#2d3139"),
                        font=dict(color="#d0d0e0"),
                        margin=dict(l=80, r=80, t=20, b=40),
                    )
                    st.plotly_chart(fig_my_shots, use_container_width=True)

                    # Shot location on court
                    st.subheader("Where I hit from")
                    _loc_fig = padel_court_heatmap_kde(
                        [h["player_x"] for h in _my_hits],
                        [h["player_y"] for h in _my_hits],
                        player_half="top" if _me in (1, 2) else "bottom",
                        width=360,
                        title="Hit locations",
                    )
                    st.plotly_chart(_loc_fig, use_container_width=False)

        # ═══════════════════════════════════════════════════════════════════
        # TAB: MY PARTNER
        # ═══════════════════════════════════════════════════════════════════
        with _tab_partner:
            st.subheader(f"{_my_name} & {_partner_name}")

            _pair_key = "1_2" if _me in (1, 2) else "3_4"
            _sync_cache = (_analytics.get("partner_synchrony") or {}).get(_pair_key)
            if _sync_cache:
                _vsync = _sync_cache.get("vertical_sync")
                _hsync = _sync_cache.get("horizontal_sync")
                _fwidth = _sync_cache.get("avg_formation_width")
            else:
                _sync_full = partner_synchrony(df, _me, _partner_id)
                _vsync = _sync_full["vertical_sync"]
                _hsync = _sync_full["horizontal_sync"]
                _fwidth = _sync_full["avg_formation_width"]

            _ps1, _ps2, _ps3 = st.columns(3)
            _vval, _vlab = _sync_label(_vsync)
            _hval, _hlab = _sync_label(_hsync)
            with _ps1:
                st.metric("Forward/back sync", _vval, help="Pearson r of Vy — how in step you are moving up/down the court")
                st.caption(_vlab)
            with _ps2:
                st.metric("Side-to-side sync", _hval, help="Pearson r of Vx — how in step you are laterally")
                st.caption(_hlab)
            with _ps3:
                _fw = f"{_fwidth:.1f} m" if _fwidth is not None else "N/A"
                st.metric("Formation width", _fw, help="Median lateral gap between partners")

            # Rolling sync chart in expander to reduce noise
            with st.expander("Show rolling sync over time"):
                _sync_full2 = partner_synchrony(df, _me, _partner_id)
                fig_sync = go.Figure()
                fig_sync.add_trace(go.Scatter(
                    x=df["time"], y=_sync_full2["rolling_vertical_sync"],
                    mode="lines", name="Forward/back (Vy)", line=dict(color="#6366f1"),
                ))
                fig_sync.add_trace(go.Scatter(
                    x=df["time"], y=_sync_full2["rolling_horizontal_sync"],
                    mode="lines", name="Side-to-side (Vx)", line=dict(color="#f43f5e"),
                ))
                fig_sync.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig_sync.update_layout(
                    yaxis=dict(title="Pearson r", range=[-1.1, 1.1], gridcolor="#2d3139"),
                    xaxis_title="Time (s)", height=260,
                    legend=dict(orientation="h", y=1.12),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#d0d0e0"),
                )
                st.plotly_chart(fig_sync, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════════
        # TAB: AI COACHING
        # ═══════════════════════════════════════════════════════════════════
        with _tab_ai:
            st.subheader(f"AI Coaching — {_my_name}")

            if not _has_ball_data:
                st.info("Full AI coaching requires ball data. Re-run the pipeline, then click Run AI Analysis.")

            _cached_ai = _analytics.get("ai_analysis")

            if _cached_ai:
                _ai_result = _cached_ai
                st.caption("Analysis loaded from cache.")
            else:
                _ai_result = None
                _run_ai = st.button("Run AI Analysis", help="Calls Claude API — requires ANTHROPIC_API_KEY")
                if _run_ai:
                    with st.spinner("Analysing match data with Claude..."):
                        _zone_for_ai = _analytics.get("zone_breakdown") or {
                            str(pid): zone_breakdown(df, pid) for pid in (1, 2, 3, 4)
                        }
                        _sync_for_ai = _analytics.get("partner_synchrony") or {
                            "1_2": {k: v for k, v in partner_synchrony(df, 1, 2).items()
                                    if k in ("vertical_sync", "horizontal_sync", "avg_formation_width")},
                            "3_4": {k: v for k, v in partner_synchrony(df, 3, 4).items()
                                    if k in ("vertical_sync", "horizontal_sync", "avg_formation_width")},
                        }
                        _kpi_for_ai = _analytics.get("coaching_kpis") or {
                            str(pid): coaching_kpis(df, pid) for pid in (1, 2, 3, 4)
                        }
                        _rallies_for_ai = _analytics.get("rallies") or []
                        _ai_result = analyse_rallies_with_claude(
                            rallies=_rallies_for_ai,
                            zone_data=_zone_for_ai,
                            sync_data=_sync_for_ai,
                            kpi_data=_kpi_for_ai,
                        )
                        if "error" not in _ai_result:
                            _analytics["ai_analysis"] = _ai_result
                            _ai_json = f"{RESULTS_DIR}/{_vid_id}_analytics.json" if _vid_id else None
                            if _ai_json and os.path.exists(_ai_json):
                                with open(_ai_json) as _ajf:
                                    _existing = json.load(_ajf)
                                _existing["ai_analysis"] = _ai_result
                                with open(_ai_json, "w") as _ajf:
                                    json.dump(_existing, _ajf, indent=2)
                            st.session_state["analytics"] = _analytics

            if _ai_result:
                if "error" in _ai_result:
                    st.error(_ai_result["error"])
                else:
                    # My player's feedback — prominent
                    _feedback = _ai_result.get("player_feedback", {})
                    _drills = _ai_result.get("training_drills", {})

                    st.markdown(
                        f"<div style='background:#1e293b;border:1px solid #334155;"
                        f"border-radius:10px;padding:18px 20px;margin-bottom:20px'>"
                        f"<p style='font-size:0.82rem;color:#94a3b8;margin-bottom:6px'>Coaching feedback</p>"
                        f"<p style='font-size:1rem;line-height:1.6;color:#e2e8f0'>"
                        f"{_feedback.get(str(_me), 'No feedback available.')}</p></div>",
                        unsafe_allow_html=True,
                    )

                    # Training drills
                    _my_drills = _drills.get(str(_me), [])
                    if _my_drills:
                        st.subheader("Recommended training drills")
                        for _di, _drill in enumerate(_my_drills, 1):
                            _parts = _drill.split(":", 1)
                            _dname = _parts[0].strip() if len(_parts) > 1 else f"Drill {_di}"
                            _ddesc = _parts[1].strip() if len(_parts) > 1 else _drill
                            st.markdown(
                                f"<div style='background:#172033;border-left:3px solid #6366f1;"
                                f"padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0'>"
                                f"<b style='color:#a5b4fc'>{_dname}</b>"
                                f"<p style='color:#cbd5e1;margin:4px 0 0;font-size:0.88rem'>{_ddesc}</p>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    # Overall patterns
                    with st.expander("Overall match patterns"):
                        for _pat in _ai_result.get("overall_patterns", []):
                            st.markdown(f"- {_pat}")

                    # Other players' feedback (secondary)
                    with st.expander("Other players' feedback"):
                        for _oid in (1, 2, 3, 4):
                            if _oid == _me:
                                continue
                            _oname = _pnames.get(str(_oid), f"Player {_oid}")
                            st.markdown(f"**{_oname}**")
                            st.write(_feedback.get(str(_oid), "No feedback available."))
                            _o_drills = _drills.get(str(_oid), [])
                            for _od in _o_drills:
                                st.markdown(f"  - {_od}")

        # ═══════════════════════════════════════════════════════════════════
        # TAB: MATCH OVERVIEW (all 4 players)
        # ═══════════════════════════════════════════════════════════════════
        with _tab_overview:
            st.subheader("Match Overview — All Players")

            # Summary metrics
            _ov_cols = st.columns(4)
            for _i, _pid in enumerate((1, 2, 3, 4)):
                _pn = _pnames.get(str(_pid), f"Player {_pid}")
                _d = df[f"player{_pid}_distance"].sum()
                _mv = df[f"player{_pid}_Vnorm4"].abs().max() * 3.6
                with _ov_cols[_i]:
                    st.metric(_pn, f"{_d:.0f} m")
                    st.caption(f"Max {_mv:.1f} km/h")

            # Zone breakdown — all 4 players
            st.subheader("Court zone breakdown")
            _all_zones = {pid: zone_breakdown(df, pid) for pid in (1, 2, 3, 4)}
            fig_all_zones = go.Figure()
            for _zk, _zc in [("front", "#22c55e"), ("transition", "#f59e0b"), ("back", "#ef4444")]:
                fig_all_zones.add_trace(go.Bar(
                    name=_zk.capitalize(),
                    x=[_pnames.get(str(p), f"Player {p}") for p in (1, 2, 3, 4)],
                    y=[_all_zones[p][_zk] for p in (1, 2, 3, 4)],
                    marker_color=_zc,
                ))
            fig_all_zones.update_layout(
                barmode="stack", yaxis_title="% of time",
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#2d3139"),
                font=dict(color="#d0d0e0"),
            )
            st.plotly_chart(fig_all_zones, use_container_width=True)

            # Speed zone distribution — all 4 players
            st.subheader("Speed zone distribution")
            fig_spd = go.Figure()
            _spd_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
            for (_zname, _lo, _hi), _zc in zip(SPEED_ZONES, _spd_colors):
                _pcts = []
                for _pid in (1, 2, 3, 4):
                    _spds = df[f"player{_pid}_Vnorm4"].abs() * 3.6
                    _n = max(len(_spds), 1)
                    _pcts.append(100 * ((_spds >= _lo) & (_spds < _hi)).sum() / _n)
                fig_spd.add_trace(go.Bar(
                    name=_zname,
                    x=[_pnames.get(str(p), f"Player {p}") for p in (1, 2, 3, 4)],
                    y=_pcts, marker_color=_zc,
                ))
            fig_spd.update_layout(
                barmode="stack", yaxis_title="% of time", height=300,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#2d3139"), font=dict(color="#d0d0e0"),
            )
            st.plotly_chart(fig_spd, use_container_width=True)

            # Position heatmaps — all 4
            st.subheader("Position heatmaps")
            _hm_cols = st.columns(4)
            for _i, _pid in enumerate((1, 2, 3, 4)):
                with _hm_cols[_i]:
                    _ph = "top" if _pid in (1, 2) else "bottom"
                    _hf = padel_court_heatmap_kde(
                        df[f"player{_pid}_x"], df[f"player{_pid}_y"],
                        player_half=_ph, width=220,
                        title=_pnames.get(str(_pid), f"P{_pid}"),
                    )
                    st.plotly_chart(_hf, use_container_width=False)
