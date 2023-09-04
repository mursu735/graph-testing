from dash import Dash, html
import dash_cytoscape as cyto

app = Dash(__name__)

directed_edges = [
    {'data': {'id': src+tgt, 'source': src, 'target': tgt}}
    for src, tgt in ['BA', 'BC', 'CD', 'DA']
]
print(directed_edges)

directed_elements = [{'data': {'id': id_}} for id_ in 'ABCD'] + directed_edges

print(directed_elements)

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-styling-9',
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '400px'},
        elements=directed_elements,
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
            },
            {
                'selector': '#BA',
                'style': {
                    'source-arrow-color': 'red',
                    'source-arrow-shape': 'triangle',
                    'line-color': 'red'
                }
            },
            {
                'selector': '#DA',
                'style': {
                    'target-arrow-color': 'blue',
                    'target-arrow-shape': 'vee',
                    'line-color': 'blue'
                }
            },
            {
                'selector': '#BC',
                'style': {
                    'mid-source-arrow-color': 'green',
                    'mid-source-arrow-shape': 'diamond',
                    'mid-source-arrow-fill': 'hollow',
                    'line-color': 'green',
                }
            },
            {
                'selector': '#CD',
                'style': {
                    'mid-target-arrow-color': 'black',
                    'mid-target-arrow-shape': 'circle',
                    'arrow-scale': 2,
                    'line-color': 'black',
                }
            }
        ]
    )
])

if __name__ == '__main__':
    app.run(debug=True)