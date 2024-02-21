# The code below is a modified version of the forllowing file.
# sentence-transformers: examples/unsupervised_learning/SimCSE/train_simcse_from_file.py

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import logging
import plotly
import textwrap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import helpers
from dash import Dash, dcc, html, Input, Output,callback

figs = []

files = helpers.natural_sort(os.listdir("output/GPT/similarities"))

for file in files:
    print(f"Processing file {file}")
    fig = plotly.io.read_json(f"output/GPT/similarities/{file}")
    figs.append(fig)

app = Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(list(range(1, len(figs)+1)), 1, id='demo-dropdown'),
    dcc.Graph(figure=figs[0], id='dropdown-graph')
])


@callback(
    Output('dropdown-graph', 'figure'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    return figs[value-1]


if __name__ == '__main__':
    app.run(debug=True)
