# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Location handling module.
"""

# pylint: disable=redefined-outer-name, invalid-name

import json
from datetime import datetime
import pytz
import requests
from timezonefinder import TimezoneFinder

def get_timezone(lat, lon):
    """Utility function to obtain the timezone from latitude and longitude.
    """
    # Select period during standard time
    year = datetime.now().year
    date_time = datetime(year, 1, 1)

    # Find the timezone based on the given coordinates
    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lat=lat, lng=lon)
    # Convert to pytz
    timezone = pytz.timezone(timezone_name)

    # Determine the time offset from UTC/GMT
    offset_seconds = timezone.utcoffset(date_time).total_seconds()
    offset_hours = offset_seconds / (60*60)  # Convert to hours

    return offset_hours

def get_elevation(lat, lon):
    """Utility function to obtain the surface elevation from latitude and longitude.
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

if __name__ == '__main__':
    lat = 37.87
    lon = -122.27
    print(f'Timezone (should be -8 hours): {get_timezone(lat, lon)}')
    print(f'Elevation (should be 52 meter): {get_elevation(lat, lon)}')
