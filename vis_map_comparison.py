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
    named_colors = px.colors.DEFAULT_PLOTLY_COLORS
    directed_elements = {}
    directory = "output/GPT/locations"
    files = os.listdir(directory)
    # TODO: Change all of the chapters to the new format and uncomment these
    for file in files:
        # Use the next line for debugging if needed
        #file = "28.csv"
        directed_edges= []
        elements = []
        seen_elements = []
        #seen_edges = []
        groups = {}
        df = pd.read_csv(f"{directory}/{file}", sep=";")
        #df = pd.read_csv(f"{directory}/4.csv", sep=";")
        #name = file.split(".")[0]
        # If multiple characters are in one place, create intermediate Group node where all characters point to, then have only one arrow from group to location(s)
        for idx, row in df.iterrows():
            #print(groups)
            location = row["City"] + ", " + str(row["Location"])
            if "|" in row["Person"]:
                characters = row["Person"].split("|")
                if not row["Person"] in groups:
                    cur = len(groups) + 1
                    groups[row["Person"]] = {"group": f"Group {cur}", "locations": [], "color": random.choice(named_colors)}
                groups[row["Person"]]["locations"].append(location)
                # Draw line from person to groups and from group to first location
                if len(groups[row["Person"]]["locations"]) == 1:
                    for character in characters:
                        source_id = character.replace(" ", "").replace(".", "").replace(",", "")
                        source = character.strip()
                        group_id = str(groups[row["Person"]]["group"]).replace(" ", "").replace(".", "").replace(",", "")
                        group = str(groups[row["Person"]]["group"]).strip()
                        directed_edges.append({'data': {'id': source_id + group_id, 'source': source_id, 'target': group_id}})
                        stylesheet.append({'selector': f"#{source_id + group_id}",
                        'style': {
                            'target-arrow-color': groups[row["Person"]]["color"],
                            'label': idx + 1,
                            'target-arrow-shape': 'vee',
                            'line-color': groups[row["Person"]]["color"]
                        }})
                        #seen_edges.append(f"#{source_id + group_id}")
                        if source not in seen_elements:
                            seen_elements.append(source)
                            elements.append({"id": source_id, "label": source})
                        if group not in seen_elements:
                            seen_elements.append(group)
                            elements.append({"id": group_id, "label": group})
                # Draw line from group's last location to current
                source_id = ""
                source = ""
                group_id = str(groups[row["Person"]]["group"]).replace(" ", "").replace(".", "").replace(",", "")
                if len(groups[row["Person"]]["locations"]) == 1:
                    source_id = str(groups[row["Person"]]["group"]).replace(" ", "").replace(".", "").replace(",", "")
                    source = str(groups[row["Person"]]["group"]).strip()
                else:
                    source_id = str(groups[row["Person"]]["locations"][-2]).replace(" ", "").replace(".", "").replace(",", "")
                    source = str(groups[row["Person"]]["locations"][-2]).strip()
                target_id = str(groups[row["Person"]]["locations"][-1]).replace(" ", "").replace(".", "").replace(",", "")
                target = str(groups[row["Person"]]["locations"][-1]).strip()
                if target not in seen_elements:
                    seen_elements.append(target)
                    elements.append({"id": target_id, "label": target})
                directed_edges.append({'data': {'id': source_id + target_id + group_id, 'source': source_id, 'target': target_id}})
                stylesheet.append({'selector': f"#{source_id + target_id + group_id}",
                                'style': {
                                    'curve-style': 'unbundled-bezier',
                                    'label': idx + 1,
                                    'target-arrow-color': groups[row["Person"]]["color"],
                                    'target-arrow-shape': 'vee',
                                    'line-color': groups[row["Person"]]["color"],
                                }})
            else:
                if row["Order"] == "all":
                    source_id = row["Person"].replace(" ", "").replace(".", "").replace(",", "")
                    source = row["Person"].strip()
                    target_id = str(location).replace(" ", "").replace(".", "").replace(",", "")
                    target = str(location).strip()
                    directed_edges.append({'data': {'id': source_id + target_id, 'source': source_id, 'target': target_id}})
                    if source not in seen_elements:
                        seen_elements.append(source)
                        elements.append({"id": source_id, "label": source})
                    if target not in seen_elements:
                        seen_elements.append(target)
                        elements.append({"id": target_id, "label": target})
                    #print(f"Adding edge to chapter {name} {source}->{target}, position is same all text")
                    #print(f"Adding edge {source}->{target}, position is same all text")
                    stylesheet.append({'selector': f"#{source_id + target_id}",
                            'style': {
                                'target-arrow-color': 'blue',
                                'target-arrow-shape': 'vee',
                                'line-color': 'blue'
                            }})
                else:
                    source_id = row["Person"].replace(" ", "").replace(".", "").replace(",", "")
                    source = row["Person"].strip()
                    target_id = str(location).replace(" ", "").replace(".", "").replace(",", "")
                    target = str(location).strip()
                    relation = row["Order"]
                    directed_edges.append({'data': {'id': source_id + target_id, 'source': source_id, 'target': target_id}})
                    #print(f"Adding edge to chapter {name} {source}->{target}, label {relation}")
                    #print(f"Adding edge {source}->{target}, label {relation}")
                    if source not in seen_elements:
                        seen_elements.append(source)
                        elements.append({"id": source_id, "label": source})
                    if target not in seen_elements:
                        seen_elements.append(target)
                        elements.append({"id":target_id, "label": target})
                    stylesheet.append({'selector': f"#{source_id + target_id}",
                            'style': {
                                'label': idx + 1,
                                'target-arrow-color': 'blue',
                                'target-arrow-shape': 'vee',
                                'line-color': 'blue'
                            }})
        
            #directed_elements = [{'data': {'id': element["id"], "label": element["label"]}} for element in elements] + directed_edges
            name = "Chapter " + file.split(".")[0]
            directed_elements[name] = [{'data': {'id': element["id"], "label": element["label"]}} for element in elements] + directed_edges
    return directed_elements

'''
files = os.listdir("output/Chapters")
for file in files:
    df = pd.read_csv(f"input/texts/{file}", sep=";")
    for idx, row in df.iterrows():       
        if len(split) == 2:
            source_id = split[0].replace(" ", "")
            source = split[0].strip()
            target_id = split[1].replace(" ", "")
            target = split[1].strip()
            if source not in secondary_edges:
                secondary_edges[source] = []
            secondary_edges[source].append({'data': {'id': source_id + target_id, 'source': source, 'target': target}})
            if source not in secondary_elements:
                secondary_elements[source] = []
            if target not in secondary_elements:
                secondary_elements[source].append(target)
            stylesheet.append({'selector': f"#{source_id + target_id}",
                'style': {
                    'line-color': 'blue'
                }})
        
        if len(row["pos"]) == "all":
            for i in range(2, len(split)):
                # Look for names in ENTITY 2
                doc = nlp(line)
                entity = split[i].strip()
                should_add_to_main = True
                if " " in entity or "'" in entity:
                    #print(f"Delimiter found in string {entity}")
                    split_list = re.split(r" |'", entity)
                    #print(my_list)
                    for word in split_list:
                        for target in doc:
                            # Need to replace all delimiters from the string processed by spaCy, as they aren't included in the regex split
                            if word == target.text.replace("'", ""):
                                if not target.pos_ == "PROPN":
                                    print(f"Word {target.text} was not a proper noun, skipping '{entity}'")
                                    should_add_to_main = False
                                    break
                else:
                    for target in doc:
                            if entity == target.text:
                                if not target.pos_ == "PROPN":
                                    print(f"Word {target.text} was not a proper noun, skipping '{entity}'")
                                    should_add_to_main = False
                                    break
            
            should_add_to_main = True
            print(f"From: {split[0].strip()}, to {split[i].strip()}, label: {split[1].strip()}")
            if should_add_to_main:
                source_id = split[0].replace(" ", "")
                source = split[0].strip()
                target_id = split[i].replace(" ", "")
                target = split[i].strip()
                relation = split[1].strip()
                directed_edges.append({'data': {'id': source_id + target_id, 'source': source, 'target': target}})
                print(f"Adding edge {source}->{split[i].strip()}, label {relation}")
                if source not in elements:
                    elements.append(source)
                if target not in elements:
                    elements.append(target)
                stylesheet.append({'selector': f"#{source_id + target_id}",
                        'style': {
                            'label': relation,
                            'target-arrow-color': 'blue',
                            'target-arrow-shape': 'vee',
                            'line-color': 'blue'
                        }})
            # If ENTITY 2 is not a proper noun, add it to the secondary graph
            else:
                source_id = split[0].replace(" ", "")
                source = split[0].strip()
                target_id = split[i].replace(" ", "")
                target = split[i].strip()
                relation = split[1].strip()
                if source not in secondary_edges:
                    secondary_edges[source] = []
                secondary_edges[source].append({'data': {'id': source_id + target_id, 'source': source, 'target': target}})
                if source not in secondary_elements:
                    secondary_elements[source] = []
                if target not in secondary_elements:
                    secondary_elements[source].append(target)
                                    
                stylesheet.append({'selector': f"#{source_id + target_id}",
                        'style': {
                            'label': relation,
                            'target-arrow-color': 'blue',
                            'target-arrow-shape': 'vee',
                            'line-color': 'blue'
                        }})              
directed_elements = [{'data': {'id': id_}} for id_ in elements] + directed_edges
'''

#print(secondary_elements)
#print(secondary_edges)



def create_map():
    locations = {}
    options = ["GPT", "Manual"]
    names = ["Fogg"]
    # Only select files that have been manually created
    directory = "output/Manual/locations"
    files = os.listdir(directory)
    files = helpers.natural_sort(files)
    print(files)
    # "Aouda", "Cromarty", "Proctor"]
    for mode in options:
        locations[mode] = {}
        for name in names:
            locations[mode][name] = []
    for mode in options:
        for file in files:
            dir = f"output/{mode}/locations"
            df = pd.read_csv(f"{dir}/{file}", sep=";")
            df = df.dropna()
            df = df.astype({'Latitude':'float','Longitude':'float'})
            df["Label"] = "Chapter " + file.split(".")[0] + ", " + df["City"]
            for name in names:
                filtered = df.loc[df['Person'].str.contains(name)]
                locations[mode][name].append(filtered)

    journeys = {}
    for mode in options:
        journeys[mode] = {}
        for name in names:
        #print(locations)
            df = pd.concat(locations[mode][name])
            #print(df)
            df = df.reset_index()
            #df2 = pd.DataFrame(np.random.uniform(-0.008,0.008,size=(df.shape[0], 2)), columns=['lat', 'long'])
            #print(df["Latitude"])
            df = df.reindex()
            #print(df)
            #df["Latitude"] = df["Latitude"] + df2["lat"]
            #df["Longitude"] = df["Longitude"] + df2["long"]
            journeys[mode][name] = df
            #journeys[name].to_csv(f"{name}.csv")
    fig = go.Figure()
    for mode in options:
        for name in names:
            fig.add_trace(go.Scattergeo(
                lat = journeys[mode][name]["Latitude"],
                lon = journeys[mode][name]["Longitude"],
                mode = 'lines',
                line = dict(width = 2),
                name = mode
            ))
    fig.add_trace(go.Scattergeo(
        lat = journeys["GPT"]["Fogg"]["Latitude"],
        lon = journeys["GPT"]["Fogg"]["Longitude"],
        hoverinfo = 'text',
        text = journeys["GPT"]["Fogg"]["Label"],
        mode = 'markers',
        name = "Cities, GPT",
        marker = dict(
            color = 'blue'
        ),       
    ))
    fig.add_trace(go.Scattergeo(
        lat = journeys["Manual"]["Fogg"]["Latitude"],
        lon = journeys["Manual"]["Fogg"]["Longitude"],
        hoverinfo = 'text',
        text = journeys["Manual"]["Fogg"]["Label"],
        mode = 'markers',
        name = "Cities, Manual",
        marker = dict(
            color = 'red'
        ),       
    ))
    fig.update_geos(
    visible=False, resolution=50,
    showcountries=True, countrycolor="RebeccaPurple"
    )
    # display DataFrame
    return fig

map = create_map()
directed_elements = create_location_graphs()
#print(directed_elements)
#print(stylesheet)
app.layout = html.Div([
    dcc.Graph(id="map", figure=map),
    html.P("Dash Cytoscape:"),
    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'cose'},
        elements=directed_elements["Chapter 28"],
        stylesheet=stylesheet
    ),
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
    Output('cytoscape', 'elements'),
    Output('click-data', 'children'),
    Input('map', 'clickData'),
    State('cytoscape', 'elements'))
def display_click_data(clickData, elements):
    if clickData:
        text = clickData["points"][0]["text"].split(",")[0]
        summary = ""
        number = text.split(" ")[1]
        with open(f"output/GPT/summary/{number}.txt") as file:
            summary = file.read()
        summary = summary.replace(". ", ".\n")
        return directed_elements[text], summary
    return elements, ""


if __name__ == '__main__':
    app.run(debug=True)
