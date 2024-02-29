import pygraphviz as pgv
from collections import Counter
import random
import time
import helpers
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from z3 import *


use_new_sorting = True

def Abs(x):
    return If(x >= 0, x, -x)

def create_path_ranks(characters, pos, graph):
    keys = list(set(pos.keys()) - set(characters))
    place_positions = {k:pos[k] for k in keys}
    #print(place_positions)
    values = place_positions.values()
    # Needed for padding the path of characters that do not go to every node
    individual_x = sorted(set(coord[0] for coord in values))
    # Higher node in graph should point to higher rank
    individual_y = sorted(set(coord[1] for coord in values), reverse=True)
    #print(individual_x)
    #print(individual_y)
    y_rank = {j:i+1 for i,j in enumerate(individual_y)}
    x_rank = {j:i+1 for i,j in enumerate(individual_x)}
    #print(x_rank)
    pos_y_rank = {}
    pos_x_rank = {}
    for place, coord in place_positions.items():
        pos_y_rank[place] = y_rank[coord[1]]
        pos_x_rank[place] = x_rank[coord[0]]
    #print("x:", pos_x_rank)
    #print("y:", pos_y_rank)
    # Recreate the path of each character and pad it to the same length for all
    character_paths = {key: [] for key in characters}
    #for person in characters:
        #print(person)
    edges_visited = []
    for edge in graph.edges():
        if edge not in edges_visited:
            edges_visited.append(edge)
            edge_data = graph.get_edge_data(edge[0], edge[1])
            #print(edge_data)
            for i, data in edge_data.items():
                #print(edge[0], "->", edge[1], data)
                #print(data["person"])
                character_paths[data["person"]].append(edge[1])
    #print(character_paths)
    def sort_func(e):
        return pos_x_rank[e]
    
    for person in character_paths:
        new_locs = sorted(character_paths[person], key=sort_func)
        character_paths[person] = new_locs

    x_ranks = {}
    y_ranks = {}
    for person, locations in character_paths.items():
        x_ranks[person] = list(map(lambda x: pos_x_rank[x], locations))
        y_ranks[person] = list(map(lambda x: pos_y_rank[x], locations))
    #print(x_ranks)
    
    # Determine which locations should be padded for each character
    for person in characters:
        missing = sorted(set(pos_x_rank.values()).difference(set(x_ranks[person])))
        print(f"Missing for {person}: {missing}")
        first = min(x_ranks[person])
        last = max(x_ranks[person])
        print(first, last)
        for value in missing:
            #print(x_ranks[person])
            # If in the beginning or end, set it to -1
            if value < first or value > last:
                y_pad = -1
            # Otherwise get the previous value
            else:
               # print(value)
                y_pad = y_ranks[person][value - 2]
            #print("Adding value", value)
            x_ranks[person].insert(value - 1, value)
            y_ranks[person].insert(value - 1, y_pad)
    #for person in x_ranks:
    #   print(person, len(x_ranks[person]))
    #print("!!!X:", x_ranks)
    #print(y_ranks)
    return x_ranks, y_ranks

def get_positions(characters, pos, graph):
    # Each character can be in one and only one position
    # All positions must be filled
    # Character must go through a specific nodes
    # Minimize the number of edge crossings
    x_ranks, y_ranks = create_path_ranks(characters, pos, graph)
    print("Calculating character rankings with z3")
    solver = Optimize()
    n_vertices = len(characters)

    # Assignment of vertices to vertical positions and rank within edges in the drawing
    # `Assignment` is represented by character name
    Assignment = []
    conversion_map = {}
    string_to_z3_var = {}
    print(characters)
    for person in characters:
        as_variable = Int(person)
        Assignment.append(as_variable)
        conversion_map[as_variable] = person
        string_to_z3_var[person] = as_variable

    
    for Person in Assignment:
        solver.add(Person >= 1)
        solver.add(Person <= n_vertices)
        #person = conversion_map[Person]

    solver.add(Distinct(Assignment))
    #print(Assignment)
    
    edges = []
    for i in range(len(Assignment)):
        first = Assignment[i]
        #print("First:", first, len(y_ranks[conversion_map[first]]))
        for j in range(i+1, len(Assignment)):
            second = Assignment[j]
            #print("Second:", second, len(y_ranks[conversion_map[second]]))
            for path in range(len(y_ranks[conversion_map[first]])):
                first_path = y_ranks[conversion_map[first]][path]
                second_path = y_ranks[conversion_map[second]][path]
                # And(first > second, first_path < second_path)
                #if (first_path < second_path):
                #edges.append(If(And(not first_path == -1, not second_path == -1, And(first > second, first_path < second_path)), 1, 0))
                # If the characters cross, add penalty
                edges.append(If(And(first > second, first_path < second_path), 1, 0))
                #edges.append(If(first > second, 1, 0))
                # TODO: May need to improve this
    #print(edges)
    # Penalize characters in the same cluster being far away
    # NOTE: The solver is MUCH faster, if the problem can be formulated as a binary statement.
    # E.g. having the below problem be simply Abs(first - second) will get the solver stuck 
    clusters = helpers.get_clusters()
    cluster_list = {}
    for character in clusters:
        if character in characters:
            #print(character)
            cluster_num = clusters[character]
            if cluster_num not in cluster_list:
                cluster_list[cluster_num] = []
            cluster_list[cluster_num].append(character)
    #print(conversion_map)
    
    for cluster in cluster_list:
        characters = cluster_list[cluster]
        if len(characters) > 1:
            for i in range(0, len(characters)):
                first = string_to_z3_var[characters[i]]
                for j in range(i+1, len(characters)):
                    second = string_to_z3_var[characters[j]]
                    edges.append(If(Abs(first - second) > len(characters), 1, 0))
                    #print(characters[i], characters[j])
    #Crossings = Sum([ Abs(character - level) for character in Assignment for level in y_ranks[conversion_map[character]] ])
    Crossings = Sum(edges)
    #print(Crossings)
    solver.minimize(Crossings)
    solver.check()
    #print(solver.model())
    answer = solver.model()
    result = {}
    for person in Assignment:
        result[conversion_map[person]] = answer[person]
    in_order = sorted(result.items(), key=lambda x:x[1].as_long())
    print("Results:", in_order)
    return result


def update_character_locations(pos, ranking):
    new_positions = {}
    #print("Positions before:")
    #print(pos)
    def sort_func(e):
        return ranking[e].as_long()
    for character in ranking:
        x, y = pos[character]
        if x not in new_positions:
            new_positions[x] = {'y' : [], 'characters': []}
        new_positions[x]['y'].append(y)
        new_positions[x]['characters'].append(character)

    for x in new_positions:
        new_positions[x]['y'] = sorted(new_positions[x]['y'], reverse=True)
        new_positions[x]['characters'] = sorted(new_positions[x]['characters'], key=sort_func)
        for i in range(len(new_positions[x]['characters'])):
            character = new_positions[x]['characters'][i]
            pos[character] = (x, new_positions[x]['y'][i])
    #print("Positions after:")
    #print(pos)

#G = pgv.AGraph(strict=False)
def generate_graph(path):
    G = nx.MultiDiGraph()

    df = pd.read_csv(f"{path}", sep=";")

    people_and_locations = {}

    people_list = {}
    locations = []
    people_ending_in_location = {}
    people_starting_in_location = {}
    '''
    with open("colors.txt") as file:
        named_colors = file.read()
        named_colors = named_colors.replace("\n", " ").split(", ")
    '''
    named_colors = px.colors.DEFAULT_PLOTLY_COLORS
    used_colors = []
    aliases = helpers.get_aliases()
    for idx, row in df.iterrows():
        target = row["City"] + ", " + str(row["Location"]) + "_" + path.split("/")[-1] # + str(row["Chapter"])
        if target not in locations:
            G.add_node(target, shape="rect")
            locations.append(target)
            people_ending_in_location[target] = [] 
            people_starting_in_location[target] = [] 
        people = row["Person"].split("|")
        for person in people:
            if person in aliases:
                person = aliases[person]
            people_ending_in_location[target].append(person)
            # First time person was seen, choose color and add to list
            if person not in people_list:
                col = random.choice(named_colors)
                while col in used_colors:
                    col = random.choice(named_colors)
                    # Reset in case there are too many characters for the colors
                    if len(used_colors) > (len(named_colors) - 2):
                        used_colors = []
                    
                used_colors.append(col)
                G.add_node(person, shape="circle", color=col)
                people_list[person] = col
            
            # First location person was in, add edge from person to first node
            if person not in people_and_locations:
                people_and_locations[person] = []
                people_starting_in_location[person] = [] 
                people_starting_in_location[person].append(person)
                G.add_edge(person, target, person=person, color=people_list[person])
            # Add edge from last location to current location
            else:
                G.add_edge(people_and_locations[person][-1], target, person=person, color=people_list[person])
                people_starting_in_location[people_and_locations[person][-1]].append(person)
            people_and_locations[person].append(target)
    
    #with open("x_people.txt", "w") as file:
    #    file.write(str(people_and_locations))
    #G.add_node("a")  # adds node 'a'

    #G.add_edge("b", "c")  # adds edge 'b'-'c' (and also nodes 'b', 'c')

    #G.layout()´

    #print(people_ending_in_location)

    #pos = nx.spring_layout(G)
    pos = nx.drawing.nx_agraph.pygraphviz_layout(
        G,
        prog='dot',
        args='-Grankdir=LR' + ' ' + '-Gordering=in'
    )

    #nx.draw_networkx(G, pos=pos, with_labels = True)
    
    #nx.drawing.nx_agraph.write_dot(G, "network.dot")
    #plt.show()
    print("Calculating character positions with z3")
    if (len(people_list) > 1):
        result = get_positions(list(people_list.keys()), pos, G)

        update_character_locations(pos, result)
    
    # Define the location shapes manually to make sure that the edges start and end nicely, (could maybe be done with backoff, investigate? (might make the different locations a pain...))
    location_shapes = {}
    size_x = 50
    size_y = 50

    for loc in locations:
        location_shapes[loc] = {'x0': pos[loc][0] - (size_x/2), 'x1': pos[loc][0] + (size_x/2), 'y0': pos[loc][1] - (size_y/2), 'y1': pos[loc][1] + (size_y/2)}

    #print(location_shapes)

    #G.draw("graph.png")
    #print(f"Layout data: {pos}")

    # Check the "level" of each character. This is done to make sure that there are as few edge crossings as possible
    # The level marks the y-coordinate of where each edge should end and the start of the next edge if applicable
    # Could z3 be used to determine the order in which the nodes should be placed to minimize crossing?
    # Alternative: If more than one character starts at one point, sort them based on the highest y-coordinate?

    # Handle the location of start point and end point separately for each node

    # Calculate where the edge of each character should be relative to each other
    def sort_func(e):
        if use_new_sorting and len(people_list) > 1:
            return result[e].as_long()
        else:
            return pos[e]

    # Ranks for each character, this determines the y-coordinate of the node
    #ranks = sorted(people_list, key=sort_func, reverse=True)
    #print(f"Sorted list: {ranks}")

    edge_starts = [i[0] for i in G.edges()]
    edge_end = [i[1] for i in G.edges()]
    start_counts = Counter(edge_starts)
    end_counts = Counter(edge_end)

    #print(start_counts)
    #print(end_counts)

    edge_x = {}
    edge_y = {}
    markers = {}
    edges_seen = {}
    #print(G.nodes.data())
    #print(f"!!!edges:\n {G.edges()}")
    for edge in G.edges():
        if edge not in edges_seen:
            edges_seen[edge] = 0
        else:
            edges_seen[edge] += 1
        edge_data = G.get_edge_data(edge[0], edge[1])
        people_in_edge_start = people_starting_in_location[edge[0]]
        people_in_edge_end = people_ending_in_location[edge[1]]
        # NOTE: With reverse=True: old sorting based on location
        # Without reverse=True: new sorting based only on z3 optimizer
        if use_new_sorting:
            end_edge_order = sorted(people_in_edge_end, key=sort_func)#, reverse=True)
            start_edge_order = sorted(people_in_edge_start, key=sort_func)#, reverse=True)
        else:
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
        point1x, point1y = (x0 + 0.25*dx, y0 + 0.05*dy)
        point2x, point2y = (x0 + 0.75*dx, y1 - 0.05*dy) 
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
        line=dict(width=1, color=people_list[person]),
        line_shape='spline',
        hoverinfo='none',
        mode='lines+markers',
        marker=dict(
            symbol="arrow",
            opacity=markers[person],
            angle=90,
            size=20)
        ))

    #print(G)

    node_list = []
    for node in people_list:
        #print(node)
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
    return fig
    #node_trace.update_traces(textposition='inside')
    #node_trace.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    #fig.show()
