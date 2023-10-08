from dash import Dash, dcc, html, Input, Output, State, callback
import dash_cytoscape as cyto
import re
import json
import os
import spacy
import pandas as pd
import plotly.graph_objects as go

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
    
    directed_elements = {}
    directory = "output/Chapters"
    files = os.listdir(directory)
    #for file in files:
    directed_edges= []
    elements = []
    seen_elements = []
    #df = pd.read_csv(f"{directory}/{file}", sep=";", names=["name", "loc", "pos"])
    df = pd.read_csv(f"{directory}/1.csv", sep=";", names=["name", "loc", "pos"])
    #name = file.split(".")[0]
    for idx, row in df.iterrows():       
        if row["pos"] == "all":
            source_id = row["name"].replace(" ", "").replace(".", "").replace(",", "")
            source = row["name"].strip()
            target_id = row["loc"].replace(" ", "").replace(".", "").replace(",", "")
            target = row["loc"].strip()
            directed_edges.append({'data': {'id': source_id + target_id, 'source': source_id, 'target': target_id}})
            if source not in seen_elements:
                seen_elements.append(source)
                elements.append({"id": source_id, "label": source})
            if target not in seen_elements:
                seen_elements.append(target)
                elements.append({"id": target_id, "label": target})
            #print(f"Adding edge to chapter {name} {source}->{target}, position is same all text")
            print(f"Adding edge {source}->{target}, position is same all text")
            stylesheet.append({'selector': f"#{source_id + target_id}",
                    'style': {
                        'target-arrow-color': 'blue',
                        'target-arrow-shape': 'vee',
                        'line-color': 'blue'
                    }})
        else:
            source_id = row["name"].replace(" ", "").replace(".", "").replace(",", "")
            source = row["name"].strip()
            target_id = row["loc"].replace(" ", "").replace(".", "").replace(",", "")
            target = row["loc"].strip()
            relation = row["pos"]
            directed_edges.append({'data': {'id': source_id + target_id, 'source': source_id, 'target': target_id}})
            #print(f"Adding edge to chapter {name} {source}->{target}, label {relation}")
            print(f"Adding edge {source}->{target}, label {relation}")
            if source not in seen_elements:
                seen_elements.append(source)
                elements.append({"id": source_id, "label": source})
            if target not in seen_elements:
                seen_elements.append(target)
                elements.append({"id":target_id, "label": target})
            stylesheet.append({'selector': f"#{source_id + target_id}",
                    'style': {
                        'label': relation,
                        'target-arrow-color': 'blue',
                        'target-arrow-shape': 'vee',
                        'line-color': 'blue'
                    }})
    
    directed_elements = [{'data': {'id': element["id"], "label": element["label"]}} for element in elements] + directed_edges
    #directed_elements[name] = [{'data': {'id': id_}} for id_ in elements] + directed_edges
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
    df = pd.read_csv("input/texts/locations.csv", sep=";")
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lat = df["Latitude"],
        lon = df["Longitude"],
        mode = 'lines',
        line = dict(width = 2, color = 'blue'),
        name = "Journey"
    ))
    fig.add_trace(go.Scattergeo(
        lat = df["Latitude"],
        lon = df["Longitude"],
        hoverinfo = 'text',
        text = df['Location'],
        mode = 'markers',
        name = "Cities"
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

app.layout = html.Div([
    dcc.Graph(figure=map),
    html.P("Dash Cytoscape:"),
    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'cose'},
        elements=directed_elements,
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
    if data:
        node = data["id"]
        # Some nodes may not have elements, if that is the case, don't do anything
        if node in secondary_elements:
            clicked_elements = secondary_elements[node]
            clicked_elements.append(node)
            clicked_edges = secondary_edges[node]
            directed_elements = [{'data': {'id': id_}} for id_ in clicked_elements] + clicked_edges
            return directed_elements
    return elements
'''
if __name__ == '__main__':
    app.run(debug=True)
