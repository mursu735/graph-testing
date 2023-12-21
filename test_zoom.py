import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_cytoscape as cyto
import random
import re
import json
import helpers
import os
import math
import spacy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from gen_pygraphviz import generate_graph



def nonlinspace(start, stop, num):
    linear = np.linspace(0, 1, num)
    my_curvature = 1
    curve = 1 - np.exp(-my_curvature*linear)
    curve = curve/np.max(curve)   #  normalize between 0 and 1
    curve  = curve*(stop - start-1) + start
    return curve

app = Dash(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

x = []
y = []

for i in range(0, 100, 5):
    x.append(i)
    y.append(math.sin(i))

x_range = max(x)
lod_cutoff = x_range / 5
print(lod_cutoff)

figs = []

arr = nonlinspace(0.1, 5, 5)
print(arr)

for level in range(1, 6):
    x = []
    y = []
    step = arr[level - 1]
    for i in np.arange(0, 100, step):
        x.append(i)
        y.append(math.sin(i))
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y)],
        layout=go.Layout(
            title=go.layout.Title(text="A Figure Specified By A Graph Object")
        )
    )
    figs.append(fig)

@callback(Output('map', 'figure'),
        Output('click-data', 'children'),
        Input('map', 'relayoutData'))
def display_relayout_data(relayoutData):
    print(relayoutData)
    if relayoutData and "xaxis.range[0]" in relayoutData:
        x_min = relayoutData["xaxis.range[0]"]
        x_max = relayoutData["xaxis.range[1]"]
        x_delta = x_max - x_min
        print("Delta:", x_delta, "Division:", lod_cutoff)
        lod_level = math.floor(x_delta / lod_cutoff)
        print(x_delta, "level:", lod_level)
        relayoutData["x_delta"] = x_delta
        relayoutData["level_of_detail"] = lod_level
        figure = figs[lod_level]
        figure['layout']['xaxis'] = {'range': (x_min, x_max)}
        if "yaxis.range[0]" in relayoutData:
            figure['layout']['yaxis'] = {'range': (relayoutData["yaxis.range[0]"], relayoutData["yaxis.range[1]"])}
        return figure, json.dumps(relayoutData, indent=2)
    if relayoutData and "xaxis.autorange" in relayoutData:
        return figs[-1], json.dumps(relayoutData, indent=2)
    return dash.no_update, json.dumps(relayoutData, indent=2)


app.layout = html.Div([
    dcc.Graph(id="map", figure=figs[-1]),
    html.Pre(id='click-data', style=styles['pre'])
])

if __name__ == '__main__':
    app.run(debug=True)
