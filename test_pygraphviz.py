import pygraphviz as pgv
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import pandas as pd

#G = pgv.AGraph(strict=False)
G = nx.MultiDiGraph()

df = pd.read_csv(f"output/Manual/locations/14.csv", sep=";")

people_and_locations = {}

people = []
locations = []

for idx, row in df.iterrows():
    target = row["City"] + ", " + str(row["Location"])
    people = row["Person"].split("|")
    for person in people:
        if person not in people_and_locations:
            people_and_locations[person] = []
            G.add_edge(person, target)
        else:
            G.add_edge(people_and_locations[person][-1], target)
        people_and_locations[person].append(target)

#G.add_node("a")  # adds node 'a'

#G.add_edge("b", "c")  # adds edge 'b'-'c' (and also nodes 'b', 'c')

#G.layout()

pos = nx.spring_layout(G)

#G.draw("graph.png")

edge_x = []
edge_y = []
#print(G.nodes())
#print(G.edges())
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

print(G)

node_list = []
for node in G.nodes():
    data = {}
    x, y = pos[node]
    data["label"] = node
    data["x"] = x
    data["y"] = y
    node_list.append(data)

df = pd.DataFrame(node_list)

print(df)

print(df["label"])

node_trace = go.Scatter(
    x=df["x"], y=df["y"],
    mode='markers',
    text=df["label"],
    marker=())

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
