import os
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("output/GPT/locations_backup.csv", sep=";")

fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lat = df["Latitude"],
    lon = df["Longitude"],
    hoverinfo = 'text',
    text = df["Location"],
    mode = 'markers+lines',
    name = "Cities",

))
fig.update_geos(
visible=False, resolution=50,
showcountries=True, countrycolor="RebeccaPurple"
)

fig.show()
# display DataFrame