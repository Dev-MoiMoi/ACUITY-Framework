"""
ACUITY Framework — Haversine Proximity

Calculates geographic distance between two coordinate pairs using the
Haversine formula for great-circle distance on a sphere.
Used for proximity-based ranking of businesses relative to the user.
"""
import math

# Earth's mean radius in kilometres
EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Return the great-circle distance (km) between two lat/lon points.

    Args:
        lat1, lon1: User's coordinates (decimal degrees).
        lat2, lon2: Business's coordinates (decimal degrees).

    Returns:
        Distance in kilometres.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_KM * c
