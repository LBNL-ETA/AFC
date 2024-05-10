# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Location handling module.
"""

from datetime import datetime
import json
from timezonefinder import TimezoneFinder
import pytz
import requests

def get_timezone(latitude, longitude):
    """Utility function to obtain the timezine from latitude and longitude
    """
    # We set the date to the 1st January and not now as the off set may vary along the year
    year = datetime.now().year
    date_time = datetime(year, 1, 1)

    # Find the timezone based on the given coordinates
    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lat=latitude, lng=longitude)
    # Get the timezone object using pytz
    timezone = pytz.timezone(timezone_name)

    # Determine the offset from UTC/GMT
    offset_seconds = timezone.utcoffset(date_time).total_seconds()
    offset_hours = offset_seconds / 3600  # Convert to hours

    return int(abs(offset_hours)*15)

def get_elevation(lat, lon):
    """Utility function to obtain the elevation of a place from latitude and longitude
    """
    # Request the elevation on open elevation
    response = requests.post(
        "https://api.open-elevation.com/api/v1/lookup",
        headers={
            "Content-Type": "application/json; charset=utf-8"
        },
        data=json.dumps({"locations": [{"latitude": lat, "longitude": lon}]}),
        timeout=10
    )
    # Convert from json
    elevation = json.loads(response.content)["results"][0]["elevation"]
    return elevation
