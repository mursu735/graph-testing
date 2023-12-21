import os
import pandas as pd
from geopy.geocoders import Nominatim
import sys
import helpers

geolocator = Nominatim(user_agent="map_visualization")

def get_country(lat, lon):
    coord = str(lat) + "," + str(lon)
    location = geolocator.reverse(coord)
    print(location)
    if location is not None:
        return location.raw['address']['country_code']
    else:
        return "sea"

prefix = "output/GPT/locations"
files = helpers.natural_sort(os.listdir(prefix))

print(files)

geolocator = Nominatim(user_agent="map_visualization")

for file in files:
    df = pd.read_csv(f"{prefix}/{file}", sep=";")
    chapter_number = file.split(".")[0]
    df['Country'] = df.apply(lambda x: get_country(x['Latitude'], x['Longitude']), axis=1)
    df.to_csv(f"{prefix}/{file}", sep=";")