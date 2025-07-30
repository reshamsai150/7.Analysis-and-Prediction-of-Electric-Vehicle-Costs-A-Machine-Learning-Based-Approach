# code/ev_stations/utils.py
import requests

def geocode_address(address):
    """
    Convert address to latitude and longitude using OpenStreetMap's Nominatim API.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json"}
    response = requests.get(url, params=params)

    if response.status_code == 200 and response.json():
        location = response.json()[0]
        return float(location["lat"]), float(location["lon"])
    else:
        return None, None
