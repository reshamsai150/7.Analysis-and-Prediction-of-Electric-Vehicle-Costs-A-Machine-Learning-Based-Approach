# code/ev_stations/locator.py
import requests

API_KEY = "YOUR_OPENCHARGEMAP_API_KEY"  # get a free API key at https://openchargemap.org/site/develop/api

def get_nearby_stations(lat, lon, distance_km=10, max_results=50):
    """
    Fetch nearby EV charging stations from OpenChargeMap API.
    """
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "latitude": lat,
        "longitude": lon,
        "distance": distance_km,
        "maxresults": max_results
    }
    headers = {"X-API-Key": API_KEY}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        stations = []
        for item in data:
            stations.append({
                "title": item["AddressInfo"]["Title"],
                "lat": item["AddressInfo"]["Latitude"],
                "lon": item["AddressInfo"]["Longitude"],
                "address": item["AddressInfo"].get("AddressLine1", ""),
                "town": item["AddressInfo"].get("Town", ""),
                "state": item["AddressInfo"].get("StateOrProvince", ""),
                "postcode": item["AddressInfo"].get("Postcode", "")
            })
        return stations
    else:
        print("Error fetching data:", response.status_code)
        return []
