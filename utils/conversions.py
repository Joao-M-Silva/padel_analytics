""" Functions for pixels distance conversions """

def convert_pixel_distance_to_meters(
    pixel_distance: float,
    reference_in_meters: int,
    reference_in_pixels: int,
) -> float:
    return (
        (pixel_distance * reference_in_meters)
        /
        reference_in_pixels
    )

def convert_meters_to_pixel_distance(
    meters: float,
    reference_in_meters: int,
    reference_in_pixels: int,
) -> int:
    return int(
        (meters * reference_in_pixels)
        /
        reference_in_meters
    )