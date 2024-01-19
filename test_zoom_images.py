from dash import Dash, dcc, html, Input, Output, State, callback
from collections import Counter
from statistics import median
from PIL import Image
import dash
import random
import re
import json
import helpers
import os
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from gen_pygraphviz import get_positions, update_character_locations

use_new_sorting = True

img_size_x = 256

def generate_positions(path):
    G = nx.MultiDiGraph()

    df = pd.read_csv(f"{path}", sep=";")

    people_and_locations = {}

    people_list = {}
    locations = []
    people_ending_in_location = {}
    people_starting_in_location = {}

    named_colors = px.colors.DEFAULT_PLOTLY_COLORS
    used_colors = []
    aliases = helpers.get_aliases()
    for idx, row in df.iterrows():
        target = row["City"] + ", " + str(row["Location"]) + "/" + row["Country"] + "_" + str(row["Chapter"])
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
        args='-Grankdir=LR' + ' ' + '-Gordering=in' + " " #+ "-Gnodesep=250"
    )
    return pos
    '''
    positions = {}
    for place in pos:
        # Country, may have multiple locations in it
        if "backup" not in place:
            if "/" in place:
                chapter_and_country = place.split("/")[1]
                #print(chapter)
                if chapter_and_country not in positions:
                    positions[chapter_and_country] = {"x": [], "y": []}
                x, y = pos[place]
                positions[chapter_and_country]["x"].append(x)
                positions[chapter_and_country]["y"].append(y)
            # Person, only one position
            else:
                x, y = pos[place]
                positions[place] = {"x": x, "y": y}
    
    for location in positions:
        if type(positions[location]["x"]) == list:
            x_median = median(sorted(positions[location]["x"]))
            y_median = median(sorted(positions[location]["y"]))
            positions[location] = {"x": x_median, "y": y_median}
    
    return positions
    '''

# TODO: When combining the graphs, all graphs should use consistent colors

# Generate all locations first, then use that position information to create the flag graph
def generate_country(path):
    G = nx.MultiDiGraph()

    df = pd.read_csv(f"{path}", sep=";")

    people_and_locations = {}

    people_list = {}
    country_last_chapter = {}
    people_last_chapter = {}
    number_of_locations = {}
    location_importance = {}
    number_of_character_mentions = {}
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
        target = row["Country"]# + "_" + str(row["Chapter"])
        if target not in country_last_chapter:
            location_importance[target] = 1
            country_last_chapter[target] = row["Chapter"]
            number_of_locations[target] = 1
            G.add_node(target)
            #n = G.get_node(target)
            #n.attr["shape"] = "box"
            #n.attr["height"] = ""
            #n.attr["width"] = 250.0
            nx.set_node_attributes(G, {target: {"shape": "box"}})
            locations.append(target)
            people_ending_in_location[target] = [] 
            people_starting_in_location[target] = []
        else:  
            if number_of_locations[target] > 1:
                target = row["Country"] + "_" + str(number_of_locations[target])
            # The second visit to the country (done mainly for UK)
            if row["Chapter"] - country_last_chapter[target] > 1:
                number_of_locations[target] += 1
                target = row["Country"] + "_" + str(number_of_locations[target])
                country_last_chapter[target] = row["Chapter"]
                G.add_node(target)
                nx.set_node_attributes(G, {target: {"shape": "box"}})
                locations.append(target)
                people_ending_in_location[target] = [] 
                people_starting_in_location[target] = []
                location_importance[target] = 1
            else:
                if row["Chapter"] - country_last_chapter[target] == 1:
                    location_importance[target] += 1
                country_last_chapter[target] = row["Chapter"]
            
                 
        people = row["Person"].split("|")
        for person in people:
            if person in aliases:
                person = aliases[person]
            if person not in people_last_chapter:
                people_last_chapter[person] = row["Chapter"]
                number_of_character_mentions[person] = 1
            else:
                if number_of_character_mentions[person] > 1:
                    person = person + "_" + str(number_of_character_mentions[person])
                if row["Chapter"] - people_last_chapter[person] > 6:
                    number_of_character_mentions[person] += 1
                    person = person + "_" + str(number_of_character_mentions[person])
                    people_last_chapter[person] = row["Chapter"]
                else:
                    people_last_chapter[person] = row["Chapter"]
            
            if person not in people_ending_in_location[target]:
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
                people_and_locations[person].append(target)
            # Add edge from last location to current location, but only if it does not exist in the list
            else:
                if target not in people_and_locations[person]:
                    if person not in people_starting_in_location[people_and_locations[person][-1]]:
                        G.add_edge(people_and_locations[person][-1], target, person=person, color=people_list[person])
                        people_starting_in_location[people_and_locations[person][-1]].append(person)
                        people_and_locations[person].append(target)
            


    #print("People and locations:", people_and_locations)
    # TODO: Change flags to GPT generated images in some layout, try with just UK for now, then add the other images
    print("Location importance:", location_importance)
    landscape_aspect_ratio = 7 / 4
    portrait_aspect_ratio = 4 / 7
    dpi = 96.0
    #wimg_size_x = 256
    img_size_y = img_size_x
    #size_y = size_x * aspect_ratio
    padding = 5

    # Calculate the boundaries of the large box that surrounds the pictures
    for loc in locations:
        if "backup" not in loc:
            # Each country should have a manually written file called layout that determines how the final panel will be shaped
            with open(f"pictures/Chapters/{loc}/layout", encoding="utf-8") as file:
                text = file.readlines()
                number_of_rows = len(text)
                #print("Number of rows:", number_of_rows)
                longest_row = 0
                total_height = 0
                for line in text:
                    line = line.replace("\n", "")
                    #print(line)
                    #img_rows = math.ceil(location_importance[loc] / 2)
                    #location_shapes[loc]["rows"] = number_of_rows
                    total_width = 0
                    panels = line.split(",")
                    max_height = 0
                    for panel in panels:
                        img_height = 0
                        img_type = panel.split("/")[1]
                        # Square
                        if img_type == "s":
                            total_width += img_size_x
                            img_height = img_size_y
                        # Landscape
                        elif img_type == "l":
                            total_width += img_size_x * landscape_aspect_ratio
                            img_height = img_size_y
                        # Portrait, has the same aspect ratio, but vertical
                        else:
                            total_width += img_size_x
                            img_height = img_size_y * landscape_aspect_ratio
                        if img_height > max_height:
                            max_height = img_height
                    if total_width > longest_row:
                        longest_row = total_width
                    total_height += max_height
                    #print("Total height:", total_height)     
                    nx.set_node_attributes(G, {loc: {"width": longest_row / dpi, "height": total_height / dpi}})
        #location_shapes[loc] = {'x0': pos[loc]["x"], 'x1': pos[loc]["x"] + (size_x), 'y0': pos[loc]["y"], 'y1': pos[loc]["y"] + (size_y ), 'image': loc.split("_")[0]}


    #pos = nx.spring_layout(G)
    # 16:9 ratio. graphviz ratio is height / width unlike traditional aspect ratio
    ratio = 1.0/1.78
    # What the window most likely will look like
    #ratio = 8.5
    pos = nx.drawing.nx_agraph.pygraphviz_layout(
        G,
        prog='dot',
        args='-Grankdir=LR' + ' ' + '-Gordering=in' + " " + f"-Gratio={ratio}" + " " + "-Gnodesep=5.0"
    )
    #print(pos)

    result = get_positions(list(people_list.keys()), pos, G)

    update_character_locations(pos, result)
    print("Updated character positions")
    #pos_tuple = {}
    #for loc in pos:
    #    pos_tuple[loc] = (pos[loc]["x"], pos[loc]["y"])
    print(pos)
    #nx.draw_networkx(G, pos=pos, with_labels = True)
    
    #nx.drawing.nx_agraph.write_dot(G, "network.dot")
    #plt.show()
    '''
    if (len(people_list) > 1):
        result = get_positions(list(people_list.keys()), pos, G)

        update_character_locations(pos, result)
    '''
    # Define the location shapes manually to make sure that the edges start and end nicely, (could maybe be done with backoff, investigate? (might make the different locations a pain...))
    location_shapes = {}
    
    size_x = 500
    
    scale_x = 10
    scale_y = 1
    max_x = 0
    max_y = 0
    for loc in pos:
        pos[loc] = {"x": pos[loc][0], "y": pos[loc][1]}
        if pos[loc]["x"] > max_x:
            max_x = pos[loc]["x"]
        if pos[loc]["y"] > max_y:
            max_y = pos[loc]["y"]
    print(f"Maximum: x: {max_x}, y: {max_y}")
    aspect_ratio = max_x / max_y
    print("Aspect ratio:", aspect_ratio)


    landscape_aspect_ratio = 7 / 4
    portrait_aspect_ratio = 4 / 7
    #img_size_x = 1024
    img_size_y = img_size_x# / aspect_ratio
    #size_y = img_size_x# * aspect_ratio
    print(f"Image size: x: {img_size_x}, y: {img_size_y}")
    padding = 5

    #print(pos)

    for loc in locations:
        if "backup" not in loc:
            #location_shapes[loc] = {'x0': pos[loc]["x"] - padding, 'x1': pos[loc]["x"] + (size_x + (scale_x * location_importance[loc])), 'y0': pos[loc]["y"] - padding, 'y1': pos[loc]["y"] + (size_y + (scale_y * location_importance[loc])), 'image': loc.split("_")[0]}
            location_shapes[loc] = {'x0': pos[loc]["x"] - padding, 'y0': pos[loc]["y"] - padding, 'image': loc.split("_")[0]}
            # Calculate the boundaries of the large box that surrounds the pictures
            #if loc == "gb":
                #location_shapes[loc]["test"] = True
            # Each country should have a manually written file called layout that determines how the final panel will be shaped
            with open(f"pictures/Chapters/{loc}/layout", encoding="utf-8") as file:
                text = file.readlines()
                number_of_rows = len(text)
                #print("Number of rows:", number_of_rows)
                longest_row = 0
                total_height = 0
                for line in text:
                    line = line.replace("\n", "")
                    #print(line)
                    #img_rows = math.ceil(location_importance[loc] / 2)
                    location_shapes[loc]["rows"] = number_of_rows
                    total_width = 0
                    panels = line.split(",")
                    max_height = 0
                    for panel in panels:
                        img_height = 0
                        img_type = panel.split("/")[1]
                        # Square
                        if img_type == "s":
                            total_width += img_size_x
                            img_height = img_size_y
                        # Landscape
                        elif img_type == "l":
                            total_width += img_size_x * landscape_aspect_ratio
                            img_height = img_size_y
                        # Portrait, has the same aspect ratio, but vertical
                        else:
                            total_width += img_size_x
                            img_height = img_size_y * landscape_aspect_ratio
                        if img_height > max_height:
                            max_height = img_height
                    if total_width > longest_row:
                        longest_row = total_width
                    total_height += max_height
                    #print("Total height:", total_height)     
                location_shapes[loc]["x1"] = pos[loc]["x"] + (longest_row) + padding
                location_shapes[loc]["y1"] = pos[loc]["y"] + (total_height) + padding
            #location_shapes[loc] = {'x0': pos[loc]["x"], 'x1': pos[loc]["x"] + (size_x), 'y0': pos[loc]["y"], 'y1': pos[loc]["y"] + (size_y ), 'image': loc.split("_")[0]}

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
        if use_new_sorting:
            return result[e].as_long()
        else:
            return (pos[e]["x"], pos[e]["y"])

    # Ranks for each character, this determines the y-coordinate of the node
    #print(people_list)
    ranks = sorted(people_list, key=sort_func, reverse=True)
    #print(f"Sorted list: {ranks}")

    edge_starts = [i[0] for i in G.edges()]
    edge_end = [i[1] for i in G.edges()]
    start_counts = Counter(edge_starts)
    end_counts = Counter(edge_end)

    #print(start_counts)
    #print(end_counts)

    edge_x = {}
    edge_y = {}
    label_data = {}
    markers = {}
    edges_seen = {}

    character_clusters = helpers.get_clusters()
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
        # Cluster ranking is based on the character with the highest ranking in the cluster
        clusters_in_edge_start = []
        [clusters_in_edge_start.append(character_clusters[x]) for x in start_edge_order if character_clusters[x] not in clusters_in_edge_start]
        clusters_in_edge_end = []
        [clusters_in_edge_end.append(character_clusters[x]) for x in end_edge_order if character_clusters[x] not in clusters_in_edge_end]
        
        #print(f"Start edge: {edge[0]}, people in it: {start_edge_order}, clusters in it: {clusters_in_edge_start}")
        #print(f"End edge: {edge[1]}, people in it: {end_edge_order}, clusters in it: {clusters_in_edge_end}")
        #print(edge_order)
        #print(len(edge_data))
        #print(f"Edge {edge}, {edge_data}")
        person = edge_data[edges_seen[edge]]
        person = person["person"]
        if person not in edge_x:
            edge_x[person] = []
            edge_y[person] = []
            markers[person] = []
            label_data[person] = []
        # Edge starts from a person so no need to do the complex calculations
        #print(edge[0])
        if edge[0] in people_list:
            #print(pos[edge[0]])
            x0, y0 = pos[edge[0]]["x"], pos[edge[0]]["y"]
        else:
            #print(location_shapes[edge[0]])
            x0, start_y = location_shapes[edge[0]]['x1'], location_shapes[edge[0]]['y1']
            number_of_edges_start = start_counts[edge[0]] + 1
            #start_gap = size_y
            #if "test" in location_shapes[edge[0]]:
            #start_gap = location_shapes[edge[0]]["rows"] * img_size_y
            start_gap = location_shapes[edge[0]]["y1"] - location_shapes[edge[0]]["y0"]
            start_increment = start_gap / number_of_edges_start
            start_location = start_edge_order.index(person) + 1
            y0 = start_y - (start_location * start_increment)
        # Get the rank of person, determine where the edge should end along the shape y axis
        x1, end_y = location_shapes[edge[1]]['x0'], location_shapes[edge[1]]['y1']
        # Determine into how many pieces the edge should be divided 
        #number_of_edges_end = end_counts[edge[1]] + 1
        number_of_edges_end = len(clusters_in_edge_end) + 1
        #print("Number of edges ending", number_of_edges_end)
        #end_gap = size_y
        #if "test" in location_shapes[edge[1]]:
        end_gap = location_shapes[edge[1]]["y1"] - location_shapes[edge[1]]["y0"]
        #end_gap = location_shapes[edge[1]]["rows"] * img_size_y
        end_increment = end_gap / number_of_edges_end
        #print(edge[1], end_increment, number_of_edges_end)
        #end_location = end_edge_order.index(person) + 1
        end_location = clusters_in_edge_end.index(character_clusters[person]) + 1
        y1 = end_y - (end_location * end_increment)
        #x1, y1 = pos[edge[1]]
        #print(f"Person: {person}, edge {edge}: {x0, x1}, {y0, y1}")
        #print(x1, x0)
        dx = x1 - x0
        dy = y1 - y0
        #x1, y1 = (x1 - 0.02*dx, y1)
        # Smoothen edges
        point1x, point1y = (x0 + 0.25*dx, y0 + 0.05*dy)
        point2x, point2y = (x0 + 0.75*dx, y1 - 0.05*dy) 
        
        edge_x[person].extend([x0, point1x, point2x, x1, None])
        edge_y[person].extend([y0, point1y, point2y, y1, None])
        # Add arrows only to the last trace
        markers[person].extend([0, 0, 0, 1, 0])
        label_text = person
        if edge[0] in helpers.country_code_to_name:
            label_text += f", start: {helpers.country_code_to_name[edge[0]]}"
        else: 
            label_text += f", start: {edge[0]}"
        if edge[1] in helpers.country_code_to_name:
            label_text += f", end: {helpers.country_code_to_name[edge[1]]}"
        else: 
            label_text += f", end: {edge[1]}"
        label_data[person].extend([label_text] * 5)


    traces = []

    print(edge_x)

    for person in edge_x:
        traces.append(go.Scatter( 
        x=edge_x[person], y=edge_y[person],
        line=dict(width=1, color=people_list[person]),
        line_shape='spline',
        hoverinfo='text',
        customdata=label_data[person],
        hovertemplate="%{x}, %{y}, %{customdata}",
        text=person,
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
        x, y = pos[node]["x"], pos[node]["y"]
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
        marker=dict(size=30,symbol=df["shape"],color=df["color"])))


    fig = go.Figure(data=traces,
                layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    #xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    #yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    )
    # Flags and the big boxes should be the same in both levels of detail
    for loc in location_shapes:
        fig.add_trace(
            go.Scatter(
                x=[location_shapes[loc]['x0'],location_shapes[loc]['x0'],location_shapes[loc]['x1'],location_shapes[loc]['x1'],location_shapes[loc]['x0']], #x1-x1-x2-x2-x1
                y=[location_shapes[loc]['y0'],location_shapes[loc]['y1'],location_shapes[loc]['y1'],location_shapes[loc]['y0'],location_shapes[loc]['y0']], #y1-y2-y2-y1-y1
                fill="toself",
                mode='lines',
                name='',
                text=f"{loc}, x0: {location_shapes[loc]['x0']}, x1: {location_shapes[loc]['x1']}, y0: {location_shapes[loc]['y0']}, y1: {location_shapes[loc]['y1']}",
                opacity=1
            ))
        country = loc.split("_")[0]
        x_delta = location_shapes[loc]['x1'] - location_shapes[loc]['x0']
        middle = (location_shapes[loc]['x1'] + location_shapes[loc]['x0']) / 2
        country_image_size_x = x_delta / 3
        path = loc
        country_image_size_y = country_image_size_x * aspect_ratio
        country_image = Image.open(f"pictures/Chapters/{path}/{country}.png")
        fig.add_layout_image(
                        x=middle,
                        y=location_shapes[loc]['y1'],
                        source=country_image,
                        xref="x",
                        yref="y",
                        sizex=country_image_size_x,
                        sizey=country_image_size_y,
                        xanchor="center",
                        yanchor="bottom",
                    )
    return fig, location_shapes, aspect_ratio, max_x

def add_images(fig, location_shapes, aspect_ratio):
    #img_size_x = 128
    img_size_y = img_size_x# / aspect_ratio
    landscape_aspect_ratio = 7 / 4
    padding = 5
    
    for loc in location_shapes:
        #if "test" in location_shapes[loc]:
        #path = location_shapes[loc]['image']
        path = loc
        images = [s for s in os.listdir(f"pictures/Chapters/{path}/") if s.endswith('.png')]
        layout = ""
        with open(f"pictures/Chapters/{loc}/layout", encoding="utf-8") as file:
            layout = file.readlines()
        number_of_rows = len(layout)
        #print(images)
        row = -1
        previous_y = location_shapes[loc]['y1'] - (padding)
        #print(layout)
        # Go through layout file, add the required images
        for line in layout:
            line = line.replace("\n", "")
            largest_y_in_row = 0
            pictures = line.split(",")
            previous_x = location_shapes[loc]['x0'] + (padding)
            for i in range(len(pictures)):
                current = pictures[i]
                filename, shape = current.split("/")
                if f"{filename}.png" in images:
                    image = Image.open(f"pictures/Chapters/{path}/{filename}.png")
                    current_img_size_x = 0
                    current_img_size_y = 0
                    #print("start:", filename, "x:", current_img_size_x, "y:", current_img_size_y)
                    if shape == "s":
                        current_img_size_x = img_size_x
                        current_img_size_y = img_size_y
                    elif shape == "l":
                        current_img_size_x = img_size_x * landscape_aspect_ratio
                        current_img_size_y = img_size_y
                    else:
                        current_img_size_x = img_size_x
                        current_img_size_y = img_size_y * landscape_aspect_ratio
                    x0 = previous_x
                    #print("after:", filename, "x:", current_img_size_x, "y:", current_img_size_y)
                    x1 = x0 + current_img_size_x
                    previous_x = x1
                    y0 = previous_y
                    y1 = y0 - current_img_size_y
                    if current_img_size_y > largest_y_in_row:
                        largest_y_in_row = current_img_size_y
                    print(filename, "x0", x0, "x1", x1, "y0", y0, "y1", y1)
                    #print("Image y:", current_img_size_y)
                    fig.add_layout_image(
                        x=x0,
                        y=y0,
                        source=image,
                        xref="x",
                        yref="y",
                        sizex=current_img_size_x,
                        sizey=current_img_size_y,
                        xanchor="left",
                        yanchor="top",
                    )
                    chapter = filename.split("_")[0]
                    summary = ""
                    with open(f"output/GPT/summary/{chapter}.txt", encoding="utf-8") as file:
                        summary = file.read()
                        summary = summary.replace(". ", ".<br>")
                    country = loc
                    if country in helpers.country_code_to_name:
                        country = helpers.country_code_to_name[country]
                    fig.add_trace(
                        go.Scatter(
                            x=[x0,x0,x1,x1,x0], #x1-x1-x2-x2-x1
                            y=[y0,y1,y1,y0,y0], #y1-y2-y2-y1-y1
                            fill="toself",
                            mode='lines',
                            name='',
                            text=f"{country}, Chapter {chapter}: {summary}",
                            #text=f"{loc}, Chapter {chapter}, x0: {x0}, x1: {x1}, y0: {y0}, y1: {y1}",
                            opacity=1
                        ))
                else:
                    print(f"WARN: Image name {filename} not found, ignoring it")
            previous_y -= largest_y_in_row
        '''  
        else:
            img_path = location_shapes[loc]["image"]
            image = Image.open(f"pictures/Flags/{img_path}.png")
            #print(loc, "x:", location_shapes[loc]['x0'], location_shapes[loc]['x1'], ", y:", location_shapes[loc]['y0'], location_shapes[loc]['y1'])
            fig.add_layout_image(
                x=location_shapes[loc]['x0'] + (size_x/2),
                y=location_shapes[loc]['y0'] + (size_y/2),
                source=image,
                xref="x",
                yref="y",
                sizex=size_x,
                sizey=size_y,
                xanchor="center",
                yanchor="middle",
            )
            fig.add_trace(
                go.Scatter(
                    x=[location_shapes[loc]['x0'],location_shapes[loc]['x0'],location_shapes[loc]['x1'],location_shapes[loc]['x1'],location_shapes[loc]['x0']], #x1-x1-x2-x2-x1
                    y=[location_shapes[loc]['y0'],location_shapes[loc]['y1'],location_shapes[loc]['y1'],location_shapes[loc]['y0'],location_shapes[loc]['y0']], #y1-y2-y2-y1-y1
                    fill="toself",
                    mode='lines',
                    name='',
                    text=f"{loc}, x0: {location_shapes[loc]['x0']}, x1: {location_shapes[loc]['x1']}, y0: {location_shapes[loc]['y0']}, y1: {location_shapes[loc]['y1']}",
                    opacity=1
                ))
        '''
    return fig


def add_overall_images(fig, location_shapes, aspect_ratio):
    #img_size_x = 128
    #img_size_y = img_size_x * aspect_ratio
    #landscape_aspect_ratio = 7 / 4
    #padding = 5
    
    for loc in location_shapes:
        #if "test" in location_shapes[loc]:
        #path = location_shapes[loc]['image']
        path = loc
        images = [s for s in os.listdir(f"pictures/Chapters/{path}/") if s.endswith('.png')]
        layout = ""
        with open(f"pictures/Chapters/{loc}/layout_overview", encoding="utf-8") as file:
            layout = file.readline()
        print(images)
        fig.add_trace(
            go.Scatter(
                x=[location_shapes[loc]['x0'],location_shapes[loc]['x0'],location_shapes[loc]['x1'],location_shapes[loc]['x1'],location_shapes[loc]['x0']], #x1-x1-x2-x2-x1
                y=[location_shapes[loc]['y0'],location_shapes[loc]['y1'],location_shapes[loc]['y1'],location_shapes[loc]['y0'],location_shapes[loc]['y0']], #y1-y2-y2-y1-y1
                fill="toself",
                mode='lines',
                name='',
                text=f"{loc}, x0: {location_shapes[loc]['x0']}, x1: {location_shapes[loc]['x1']}, y0: {location_shapes[loc]['y0']}, y1: {location_shapes[loc]['y1']}",
                opacity=1
            ))
        row = -1
        #previous_y = location_shapes[loc]['y1'] - (padding)
        #print(layout)
        # Go through layout file, add the required images
        filename = layout
        if f"{filename}.png" in images:
            image = Image.open(f"pictures/Chapters/{path}/{filename}.png")
            x0, x1 = location_shapes[loc]['x0'], location_shapes[loc]['x1']
            center = (x1 + x0) / 2
            y0, y1 = location_shapes[loc]['y0'], location_shapes[loc]['y1']
            middle = (y0 + y1) / 2
            img_size_x = x1 - x0
            img_size_y = y1 - y0
            fig.add_layout_image(
                x=center,
                y=middle,
                source=image,
                xref="x",
                yref="y",
                sizex=img_size_x,
                sizey=img_size_y,
                xanchor="center",
                yanchor="middle",
            )

        else:
            print(f"WARN: Image name {filename} not found, ignoring it")
        
    return fig



#print(generate_positions("whole_book.csv"))
figs = []
base_fig, location_shapes, aspect_ratio, max_x = generate_country("whole_book.csv")
base_fig.update_layout(
    width=1920,
    height=1080)
print("Base figure generated, adding images")
detailed_fig = add_images(go.Figure(base_fig), location_shapes, aspect_ratio)
figs.append(detailed_fig)
overall_fig = add_overall_images(go.Figure(base_fig), location_shapes, aspect_ratio)
figs.append(overall_fig)
#fig.show()

run_server = True

if not run_server:
    base_fig.show()
    detailed_fig.show()
    #detailed_fig.write_html("detailed_images.html")
    overall_fig.show()
    #overall_fig.write_html("overall_images.html")
else:
    app = Dash(__name__)

    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    x_range = max_x
    lod_cutoff = x_range / 2
    print("LOD cutoff:", lod_cutoff)

    '''
    def nonlinspace(start, stop, num):
        linear = np.linspace(0, 1, num)
        my_curvature = 1
        curve = 1 - np.exp(-my_curvature*linear)
        curve = curve/np.max(curve)   #  normalize between 0 and 1
        curve  = curve*(stop - start-1) + start
        return curve



    x = []
    y = []

    for i in range(0, 100, 5):
        x.append(i)
        y.append(math.sin(i))

    #x_range = max(x)
    lod_cutoff = x_range
    print(lod_cutoff)
    figs = []

    arr = nonlinspace(0.1, 5, 5)
    print(arr)

    for level in range(1, 6):
        x = []
        y = []
        step = arr[level - 1]
        for i in np.arange(0, 100, step):
            x.append(i)
            y.append(math.sin(i))
        fig = go.Figure(
            data=[go.Scatter(x=x, y=y)],
            layout=go.Layout(
                title=go.layout.Title(text="A Figure Specified By A Graph Object")
            )
        )
        figs.append(fig)
    '''

    @callback(Output('map', 'figure'),
            Output('click-data', 'children'),
            Input('map', 'relayoutData'))
    def display_relayout_data(relayoutData):
        print(relayoutData)
        if relayoutData and "xaxis.range[0]" in relayoutData:
            x_min = relayoutData["xaxis.range[0]"]
            x_max = relayoutData["xaxis.range[1]"]
            x_delta = x_max - x_min
            print("Delta:", x_delta, "Division:", lod_cutoff)
            lod_level = math.floor(x_delta / lod_cutoff)
            print(x_delta, "level:", lod_level)
            relayoutData["x_delta"] = x_delta
            relayoutData["level_of_detail"] = lod_level
            figure = figs[lod_level]
            figure['layout']['xaxis'] = {'range': (x_min, x_max)}
            if "yaxis.range[0]" in relayoutData:
                figure['layout']['yaxis'] = {'range': (relayoutData["yaxis.range[0]"], relayoutData["yaxis.range[1]"])}
            return figure, json.dumps(relayoutData, indent=2)
        if relayoutData and "xaxis.autorange" in relayoutData:
            return figs[-1], json.dumps(relayoutData, indent=2)
        return dash.no_update, json.dumps(relayoutData, indent=2)


    app.layout = html.Div([
        dcc.Graph(id="map", figure=figs[-1]),
        html.Pre(id='click-data', style=styles['pre'])
    ])

    if __name__ == '__main__':
        app.run(debug=True)
