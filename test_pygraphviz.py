import pygraphviz as pgv
import random
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import pandas as pd

#G = pgv.AGraph(strict=False)
G = nx.MultiDiGraph()

df = pd.read_csv(f"output/Manual/locations/14.csv", sep=";")

people_and_locations = {}

people_list = {}
locations = []
named_colors = px.colors.DEFAULT_PLOTLY_COLORS

for idx, row in df.iterrows():
    target = row["City"] + ", " + str(row["Location"])
    if target not in locations:
        G.add_node(target, shape="square")
        locations.append(target)
    people = row["Person"].split("|")
    for person in people:
        # First time person was seen, choose color and add to list
        if person not in people_list:
            col = random.choice(named_colors)
            G.add_node(person, shape="circle", color=col)
            people_list[person] = col
        
        # First location person was in, add edge from person to first node
        if person not in people_and_locations:
            people_and_locations[person] = []
            G.add_edge(person, target, person=person)
        # Add edge from last location to current location
        else:
            G.add_edge(people_and_locations[person][-1], target, person=person)
        people_and_locations[person].append(target)

#G.add_node("a")  # adds node 'a'

#G.add_edge("b", "c")  # adds edge 'b'-'c' (and also nodes 'b', 'c')

#G.layout()´

#pos = nx.spring_layout(G)
pos = nx.drawing.nx_agraph.pygraphviz_layout(
        G,
        prog='dot',
        args='-Grankdir=LR'
    )

#G.draw("graph.png")


edge_x = {}
edge_y = {}
markers = {}
edges_seen = {}
print(G.nodes.data())
print(f"!!!edges:\n {G.edges()}")
for edge in G.edges():
    if edge not in edges_seen:
        edges_seen[edge] = 0
    else:
        edges_seen[edge] += 1
    edge_data = G.get_edge_data(edge[0], edge[1])
    #print(f"Edge {edge}, {asd}")
    person = edge_data[edges_seen[edge]]
    person = person["person"]
    if person not in edge_x:
        edge_x[person] = []
        edge_y[person] = []
        markers[person] = []
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    dx = x1 - x0
    dy = y1 - y0
    x1, y1 = (x1 - 0.02*dx, y1)
    point1x, point1y = (x0 + 0.25*dx, y0)
    point2x, point2y = (x0 + 0.75*dx, y1) 
    # Smoothen edges
    edge_x[person].append(x0)
    edge_x[person].append(point1x)
    edge_x[person].append(point2x)
    edge_x[person].append(x1)
    edge_x[person].append(None)
    edge_y[person].append(y0)
    edge_y[person].append(point1y)
    edge_y[person].append(point2y)
    edge_y[person].append(y1)
    edge_y[person].append(None)
    # Add arrows only to the last trace
    markers[person].append(0)
    markers[person].append(0)
    markers[person].append(0)
    markers[person].append(1)
    markers[person].append(0)


# Add trace for each person

traces = []

for person in edge_x:
    traces.append(go.Scatter( 
    x=edge_x[person], y=edge_y[person],
    line=dict(width=0.5, color=people_list[person]),
    line_shape='spline',
    hoverinfo='none',
    mode='lines+markers',
    marker=dict(
        symbol="arrow",
        opacity=markers[person],
        angleref="previous",
        size=20)
    ))

#print(G)

node_list = []
for node in G.nodes():
    #print(node)
    data = {}
    x, y = pos[node]
    data["label"] = node
    data["x"] = x
    data["y"] = y
    data["shape"] = G.nodes.data()[node]["shape"]
    data["color"] = G.nodes.data()[node].get("color", "red")
    node_list.append(data)

df = pd.DataFrame(node_list)

print(df)

print(df["label"])

traces.append(go.Scatter(
    x=df["x"], y=df["y"],
    mode='markers+text',
    text=df["label"],
    marker=dict(size=50,symbol=df["shape"],color=df["color"])))


fig = go.Figure(data=traces,
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

#node_trace.update_traces(textposition='inside')
#node_trace.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
