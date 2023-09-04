from dash import Dash, html
import dash_cytoscape as cyto

app = Dash(__name__)

directed_edges= []

elements = []

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
            directed_edges.append({'data': {'id': split[0].strip() + split[1].strip(), 'source': split[0].strip(), 'target': split[1].strip()}})
            if split[0].strip() not in elements:
                elements.append(split[0].strip())
            if split[1].strip() not in elements:
                elements.append(split[1].strip())
            stylesheet.append({'selector': f"#{split[0].strip() + split[1].strip()}",
                'style': {
                    'target-arrow-color': 'blue',
                    'target-arrow-shape': 'vee',
                    'line-color': 'blue'
                }})
        if len(split) >= 3:
            for i in range(2, len(split)):
                directed_edges.append({'data': {'id': split[0].strip() + split[i].strip(), 'source': split[0].strip(), 'target': split[i].strip()}})
                if split[0].strip() not in elements:
                    elements.append(split[0].strip())
                if split[i].strip() not in elements:
                    elements.append(split[i].strip())
                #print(f"From: {split[0].strip()}, to {split[i].strip()}, label: {split[1].strip()}")
                stylesheet.append({'selector': f"#{split[0].strip() + split[i].strip()}",
                    'style': {
                        'label': split[1].strip(),
                        'target-arrow-color': 'blue',
                        'target-arrow-shape': 'vee',
                        'line-color': 'blue'
                    }})
        

directed_elements = [{'data': {'id': id_}} for id_ in elements] + directed_edges


app.layout = html.Div([
    html.P("Dash Cytoscape:"),
    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'cose'},
        elements=directed_elements,
        stylesheet=stylesheet
    )
])


app.run_server(debug=True)
