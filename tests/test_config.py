"""Unit tests: verify config.py reads environment variables correctly."""

import os
import importlib


def _reload_config(env_overrides: dict) -> object:
    """Reload config with the given env vars set."""
    original = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update({k: str(v) for k, v in env_overrides.items()})
    try:
        import config
        importlib.reload(config)
        return config
    finally:
        # Restore original env
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_default_values():
    import config
    importlib.reload(config)
    assert config.INPUT_VIDEO_PATH == "./examples/videos/rally.mp4"
    assert config.OUTPUT_VIDEO_PATH == "results.mp4"
    assert config.MAX_FRAMES is None
    assert config.PLAYERS_TRACKER_BATCH_SIZE == 8
    assert config.BALL_TRACKER_BATCH_SIZE == 8


def test_env_override_batch_size():
    cfg = _reload_config({"PLAYERS_TRACKER_BATCH_SIZE": "4"})
    assert cfg.PLAYERS_TRACKER_BATCH_SIZE == 4


def test_env_override_output_path():
    cfg = _reload_config({"OUTPUT_VIDEO_PATH": "/tmp/custom_output.mp4"})
    assert cfg.OUTPUT_VIDEO_PATH == "/tmp/custom_output.mp4"


def test_env_override_max_frames():
    cfg = _reload_config({"MAX_FRAMES": "100"})
    assert cfg.MAX_FRAMES == 100


def test_env_collect_data_false():
    cfg = _reload_config({"COLLECT_DATA": "false"})
    assert cfg.COLLECT_DATA is False
