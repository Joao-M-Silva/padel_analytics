"""Smoke tests: verify all key modules import without error."""


def test_config_imports():
    import config  # noqa: F401


def test_estimate_velocity_imports():
    from estimate_velocity import BallVelocityEstimator, ImpactType

    assert ImpactType.FLOOR.value == "floor"
    assert ImpactType.RACKET.value == "racket"


def test_estimate_velocity_classes_exist():
    from estimate_velocity import BallVelocityData, BallVelocity, BallVelocityEstimator

    # BallVelocity instantiates correctly
    v = BallVelocity(norm=10.0, vx=6.0, vy=8.0)
    assert abs(v.norm - 10.0) < 1e-9
    assert "m/s" in str(v)



def test_utils_import():
    from utils.conversions import (  # noqa: F401
        convert_pixel_distance_to_meters,
        convert_meters_to_pixel_distance,
    )
