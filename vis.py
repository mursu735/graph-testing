from dash import Dash, dcc, html, Input, Output, State, callback
import dash_cytoscape as cyto
import re
import json
import spacy

app = Dash(__name__)

spacy.prefer_gpu()
# More accurate but slower. If this model does not work, comment this out and uncomment the next model
nlp = spacy.load("en_core_web_trf")
# Less accurate but faster and most likely does not need GPU
#nlp = spacy.load("en_core_web_sm")

# "Main" graph, contains only elements related to names
directed_edges= []
elements = []

# "Secondary" graph, contains all the other elements, linked to the relevant node 
secondary_elements = {}
secondary_edges = {}

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
                    'label': 'data(id)'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    # The default curve style does not work with certain arrows
                    'curve-style': 'bezier'
                }
            }]

with open("output.txt") as file:
    lines = file.readlines()
    for line in lines:
        line = line.replace("[", "").replace("]", "").replace('"', '').replace("\n", "")
        
        split = line.split(",")
        
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
        
        if len(split) >= 3:
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
                                    #print(f"Word {target.text} was not a proper noun, skipping '{entity}'")
                                    should_add_to_main = False
                                    break
                else:
                    for target in doc:
                            if entity == target.text:
                                if not target.pos_ == "PROPN":
                                    #print(f"Word {target.text} was not a proper noun, skipping '{entity}'")
                                    should_add_to_main = False
                                    break
                #print(f"From: {split[0].strip()}, to {split[i].strip()}, label: {split[1].strip()}")
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

#print(secondary_elements)
#print(secondary_edges)


app.layout = html.Div([
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

if __name__ == '__main__':
    app.run(debug=True)
