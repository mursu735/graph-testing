import pygraphviz as pgv
from collections import Counter
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
people_ending_in_location = {}
people_starting_in_location = {}
named_colors = px.colors.DEFAULT_PLOTLY_COLORS
used_colors = []

for idx, row in df.iterrows():
    target = row["City"] + ", " + str(row["Location"])
    if target not in locations:
        G.add_node(target, shape="square")
        locations.append(target)
        people_ending_in_location[target] = [] 
        people_starting_in_location[target] = [] 
    people = row["Person"].split("|")
    for person in people:
        people_ending_in_location[target].append(person)
        # First time person was seen, choose color and add to list
        if person not in people_list:
            col = random.choice(named_colors)
            while col in used_colors:
                col = random.choice(named_colors)
            used_colors.append(col)
            G.add_node(person, shape="circle", color=col)
            people_list[person] = col
        
        # First location person was in, add edge from person to first node
        if person not in people_and_locations:
            people_and_locations[person] = []
            people_starting_in_location[person] = [] 
            people_starting_in_location[person].append(person)
            G.add_edge(person, target, person=person)
        # Add edge from last location to current location
        else:
            G.add_edge(people_and_locations[person][-1], target, person=person)
            people_starting_in_location[people_and_locations[person][-1]].append(person)
        people_and_locations[person].append(target)

#G.add_node("a")  # adds node 'a'

#G.add_edge("b", "c")  # adds edge 'b'-'c' (and also nodes 'b', 'c')

#G.layout()´

print(people_ending_in_location)

#pos = nx.spring_layout(G)
pos = nx.drawing.nx_agraph.pygraphviz_layout(
    G,
    prog='dot',
    args='-Grankdir=LR'
)

# Define the location shapes manually to make sure that the edges start and end nicely, (could maybe be done with backoff, investigate? (might make the different locations a pain...))
location_shapes = {}
size_x = 50
size_y = 50

for loc in locations:
    location_shapes[loc] = {'x0': pos[loc][0] - (size_x/2), 'x1': pos[loc][0] + (size_x/2), 'y0': pos[loc][1] - (size_y/2), 'y1': pos[loc][1] + (size_y/2)}

print(location_shapes)

#G.draw("graph.png")
print(f"Layout data: {pos}")

# Check the "level" of each character. This is done to make sure that there are as few edge crossings as possible
# The level marks the y-coordinate of where each edge should end and the start of the next edge if applicable
# Could z3 be used to determine the order in which the nodes should be placed to minimize crossing?

# Handle the location of start point and end point separately for each node

# Calculate where the edge of each character should be relative to each other
def sort_func(e):
    return pos[e]
# Ranks for each character, this determines the y-coordinate of the node
ranks = sorted(people_list, key=sort_func, reverse=True)
print(f"Sorted list: {ranks}")

edge_starts = [i[0] for i in G.edges()]
edge_end = [i[1] for i in G.edges()]
start_counts = Counter(edge_starts)
end_counts = Counter(edge_end)

print(start_counts)
print(end_counts)

edge_x = {}
edge_y = {}
markers = {}
edges_seen = {}
#print(G.nodes.data())
print(f"!!!edges:\n {G.edges()}")
for edge in G.edges():
    if edge not in edges_seen:
        edges_seen[edge] = 0
    else:
        edges_seen[edge] += 1
    edge_data = G.get_edge_data(edge[0], edge[1])
    people_in_edge_start = people_starting_in_location[edge[0]]
    people_in_edge_end = people_ending_in_location[edge[1]]
    end_edge_order = sorted(people_in_edge_end, key=sort_func, reverse=True)
    start_edge_order = sorted(people_in_edge_start, key=sort_func, reverse=True)
    #print(f"Start edge: {edge[0]}, people in it: {start_edge_order}")
    #print(f"End edge: {edge[1]}, people in it: {end_edge_order}")
    #print(edge_order)
    #print(len(edge_data))
    #print(f"Edge {edge}, {edge_data}")
    person = edge_data[edges_seen[edge]]
    person = person["person"]
    if person not in edge_x:
        edge_x[person] = []
        edge_y[person] = []
        markers[person] = []
    # Edge starts from a person so no need to do the complex calculations
    if edge[0] in people_list:
        x0, y0 = pos[edge[0]]
    else:
        x0, start_y = location_shapes[edge[0]]['x1'], location_shapes[edge[0]]['y1']
        number_of_edges_start = start_counts[edge[0]] + 1
        start_increment = size_y / number_of_edges_start
        start_location = start_edge_order.index(person) + 1
        y0 = start_y - (start_location * start_increment)
    # Get the rank of person, determine where the edge should end along the shape y axis
    x1, end_y = location_shapes[edge[1]]['x0'], location_shapes[edge[1]]['y1']
    # Determine into how many pieces the edge should be divided 
    number_of_edges_end = end_counts[edge[1]] + 1
    end_increment = size_y / number_of_edges_end
    end_location = end_edge_order.index(person) + 1
    y1 = end_y - (end_location * end_increment)
    #x1, y1 = pos[edge[1]]
    dx = x1 - x0
    dy = y1 - y0
    #x1, y1 = (x1 - 0.02*dx, y1)
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
for node in people_list:
    print(node)
    data = {}
    x, y = pos[node]
    data["label"] = node
    data["x"] = x
    data["y"] = y
    data["shape"] = G.nodes.data()[node]["shape"]
    data["color"] = G.nodes.data()[node].get("color", "lightblue")
    node_list.append(data)

df = pd.DataFrame(node_list)

#print(df)

#print(df["label"])

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
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                )

for loc in location_shapes:
    fig.add_shape(
    type="rect",
    fillcolor="lightblue",
    x0=location_shapes[loc]['x0'],
    y0=location_shapes[loc]['y0'],
    x1=location_shapes[loc]['x1'],
    y1=location_shapes[loc]['y1'],
    label=dict(
        text=loc
    )
    )

#node_trace.update_traces(textposition='inside')
#node_trace.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
