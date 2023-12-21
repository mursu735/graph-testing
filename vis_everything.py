from dash import Dash, dcc, html, Input, Output, State, callback
import dash_cytoscape as cyto
import random
import re
import json
import helpers
import os
import spacy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from gen_pygraphviz import generate_graph

app = Dash(__name__)

spacy.prefer_gpu()
# More accurate but slower. If this model does not work, comment this out and uncomment the next model
#nlp = spacy.load("en_core_web_trf")
# Less accurate but faster and most likely does not need GPU
#nlp = spacy.load("en_core_web_sm")

# "Main" graph, contains only elements related to names
#directed_edges= []
#elements = []
#directed_elements = {}

# "Secondary" graph, contains all the other elements, linked to the relevant node 
#secondary_elements = {}
#secondary_edges = {}

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    # The default curve style does not work with certain arrows
                    'curve-style': 'bezier'
                }
            }]

def create_location_graphs():
    #named_colors = px.colors.DEFAULT_PLOTLY_COLORS
    directed_elements = {}
    directory = "output/GPT/locations"
    files = os.listdir(directory)
    #files = ["4.csv"]
    for file in files:
        # Use the next line for debugging if needed
        #file = "28.csv"
        print(f"Generating for chapter {file}")
        graph = generate_graph(f"{directory}/{file}")
        #print(graph)   
        name = "Chapter " + file.split(".")[0]
        directed_elements[name] = graph
    return directed_elements


#print(secondary_elements)
#print(secondary_edges)



def create_map():
    locations = {}
    names = ["Fix", "Passepartout", "Fogg"]
    # "Aouda", "Cromarty", "Proctor"]
    for name in names:
        locations[name] = []
    directory = "output/GPT/locations"
    files = os.listdir(directory)
    files = helpers.natural_sort(files)
    for file in files:
        df = pd.read_csv(f"{directory}/{file}", sep=";")
        df = df.dropna()
        df = df.astype({'Latitude':'float','Longitude':'float'})
        df["Label"] = "Chapter " + file.split(".")[0] + ", " + df["City"]
        for name in names:
            filtered = df.loc[df['Person'].str.contains(name)]
            locations[name].append(filtered)

    journeys = {}
    for name in names:
    #print(locations)
        df = pd.concat(locations[name])
        #print(df)
        df = df.reset_index()
        df2 = pd.DataFrame(np.random.uniform(-0.008,0.008,size=(df.shape[0], 2)), columns=['lat', 'long'])
        #print(df["Latitude"])
        df = df.reindex()
        #print(df)
        df["Latitude"] = df["Latitude"] + df2["lat"]
        df["Longitude"] = df["Longitude"] + df2["long"]
        journeys[name] = df
        journeys[name].to_csv(f"{name}.csv")
    fig = go.Figure()
    for name in names:
        if "Fogg" not in name:
            fig.add_trace(go.Scattergeo(
                lat = journeys[name]["Latitude"],
                lon = journeys[name]["Longitude"],
                mode = 'lines',
                line = dict(width = 2),
                name = name,
                hoverinfo="skip"
            ))
        else:
            fig.add_trace(go.Scattergeo(
                lat = journeys[name]["Latitude"],
                lon = journeys[name]["Longitude"],
                mode = 'lines',
                line = dict(width = 2),
                name = name
            ))
    fig.add_trace(go.Scattergeo(
        lat = journeys["Fogg"]["Latitude"],
        lon = journeys["Fogg"]["Longitude"],
        hoverinfo = 'text',
        text = journeys["Fogg"]["Label"],
        mode = 'markers',
        name = "Cities",
        marker = dict(
            color = list(range(0, journeys["Fogg"].shape[0])),
            colorscale="RdBu"
        ),       
    ))
    fig.update_geos(
    visible=False, resolution=50,
    showcountries=True, countrycolor="RebeccaPurple"
    )
    # display DataFrame
    return fig

print("Creating map")
map = create_map()
print("Creating location graphs")
directed_elements = create_location_graphs()
print("Done")
#print(directed_elements)
#print(stylesheet)
app.layout = html.Div([
    dcc.Graph(id="map", figure=map),
    html.P("Dash Cytoscape:"),
    dcc.Graph(id="cytoscape", figure=directed_elements["Chapter 1"]),
    html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns'),
    cyto.Cytoscape(
        id='detail',
        layout={'name': 'cose'},
        elements=[],
        stylesheet=stylesheet
    ),
])

'''
@callback(
    Output('detail', 'elements'),
    Input('cytoscape', 'tapNodeData'),
    State('detail', 'elements'))
def update_elements(data, elements):
    print("/////")
    print(data)
    if data:
        #node = data["id"]
        print(data)
        
        # Some nodes may not have elements, if that is the case, don't do anything
        if node in secondary_elements:
            clicked_elements = secondary_elements[node]
            clicked_elements.append(node)
            clicked_edges = secondary_edges[node]
            directed_elements = [{'data': {'id': id_}} for id_ in clicked_elements] + clicked_edges
            return directed_elements
        
    return elements
'''
@callback(
    Output('cytoscape', 'figure'),
    Output('click-data', 'children'),
    Input('map', 'clickData'),
    State('cytoscape', 'figure'))
def display_click_data(clickData, elements):
    if clickData:
        text = clickData["points"][0]["text"].split(",")[0]
        summary = "No summary found"
        number = text.split(" ")[1]
        if os.path.isfile(f"output/GPT/summary/{number}.txt"):
            with open(f"output/GPT/summary/{number}.txt", encoding="utf-8") as file:
                summary = file.read()
        summary = summary.replace(". ", ".\n")
        return directed_elements[text], summary
    return elements, ""


if __name__ == '__main__':
    app.run(debug=True)
