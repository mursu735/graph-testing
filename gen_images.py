from collections import Counter
from datetime import timedelta
from PIL import Image
import textwrap
import random

import helpers
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
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
    
    with open("colors.txt") as file:
        side_character_colors = file.read()
        side_character_colors = side_character_colors.replace("\n", " ").split(", ")
      
    main_character_colors = px.colors.DEFAULT_PLOTLY_COLORS
    main_used_colors = []
    side_used_colors = []
    aliases = helpers.get_aliases()
    aliases_reversed = helpers.get_aliases_reversed()
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
                # TODO: Have default plotly colors for important characters and the CSS list for others
                width = 1
                line_style = "dot"
                # Character was seen previously but reintroduced, keep the old color
                if person.split("_")[0] in people_list:
                    col = people_list[person.split("_")[0]]["color"]
                else:
                    # Use a different color palette for important characters
                    if os.path.isfile(f"pictures/People/{person}_face.png"):
                        width = 2
                        line_style = "solid"
                        col = random.choice(main_character_colors)
                        while col in main_used_colors:
                            print("Clash in main character color selection", col)
                            col = random.choice(main_character_colors)
                            # Reset in case there are too many characters for the colors
                            if len(main_used_colors) == (len(main_character_colors)):
                                main_used_colors = []
                            main_used_colors.append(col)
                    else:
                        col = random.choice(side_character_colors)
                        while col in side_used_colors:
                            print("Clash in side character color selection", col)
                            col = random.choice(side_character_colors)
                            # Reset in case there are too many characters for the colors
                            if len(side_used_colors) == (len(side_character_colors)):
                                side_used_colors = []
                            side_used_colors.append(col)
                

                G.add_node(person, shape="circle", color=col)
                people_list[person] = {"color": col, "width": width, "dash": line_style}
            
            # First location person was in, add edge from person to first node
            if person not in people_and_locations:
                people_and_locations[person] = []
                people_starting_in_location[person] = [] 
                people_starting_in_location[person].append(person)
                G.add_edge(person, target, person=person, color=people_list[person]["color"])
                people_and_locations[person].append(target)
            # Add edge from last location to current location, but only if it does not exist in the list
            else:
                if target not in people_and_locations[person]:
                    if person not in people_starting_in_location[people_and_locations[person][-1]]:
                        G.add_edge(people_and_locations[person][-1], target, person=person, color=people_list[person]["color"])
                        people_starting_in_location[people_and_locations[person][-1]].append(person)
                        people_and_locations[person].append(target)
            


    #print("People and locations:", people_and_locations)
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
    label_data = {}
    markers = {}
    edges_seen = {}
    
    character_clusters = helpers.get_clusters()
    character_descriptions = helpers.get_character_descriptions()

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
        [clusters_in_edge_start.append(character_clusters[x.split("_")[0]]) for x in start_edge_order if character_clusters[x.split("_")[0]] not in clusters_in_edge_start]
        clusters_in_edge_end = []
        [clusters_in_edge_end.append(character_clusters[x.split("_")[0]]) for x in end_edge_order if character_clusters[x.split("_")[0]] not in clusters_in_edge_end]
        
        #print(f"Start edge: {edge[0]}, people in it: {start_edge_order}, clusters in it: {clusters_in_edge_start}")
        #print(f"End edge: {edge[1]}, people in it: {end_edge_order}, clusters in it: {clusters_in_edge_end}")
        #print(edge_order)
        #print(len(edge_data))
        #print(f"Edge {edge}, {edge_data}")
        person = edge_data[edges_seen[edge]]
        person = person["person"].split("_")[0]
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
        # Line is between countries, so give the summary of the events that happened in the country per person
        if edge[0] in helpers.country_code_to_name:
            start_chapter = df.loc[(df["Person"].str.contains(person)) & (df["Country"] == edge[0].split('_')[0])]["Chapter"].min()
            end_chapter = df.loc[(df["Person"].str.contains(person)) & (df["Chapter"] >= start_chapter) & (df["Country"] == edge[1].split('_')[0])]["Chapter"].min()
            label_text += f", start: {helpers.country_code_to_name[edge[0]]} (chapter {start_chapter})"
            if edge[1].split("_")[0] in helpers.country_code_to_name:
                label_text += f", end: {helpers.country_code_to_name[edge[1].split('_')[0]]} (chapter {end_chapter})"
            else: 
                label_text += f", end: {edge[1]}"
            label_text += "<br>"
            # Get chapters where the location started and ended, combine the summaries
            
            #chapters = df.loc[(df["Country"] == edge[0]) & (df["Chapter"] <= chapter_cutoff) & (df["Person"].str.contains(person))]
            '''
            #if edge[0] == "ie":
            if person == "John Bunsby":
                print(f"Searching for {edge[0]} and character {person}, chapter start {start_chapter}, end {end_chapter} (country {edge[1]})")
                #print(chapters)
            '''    
            print(f"Searching for character {person}, chapter start {start_chapter} (country {edge[0]}), end {end_chapter} (country {edge[1]})")
            for i in range(start_chapter, end_chapter+1):
                if os.path.isfile(f"output/GPT/character_summaries/{i}/{person}.txt"):
                        print(f"Searching for chapter {i}, found person {person}")
                        with open(f"output/GPT/character_summaries/{i}/{person}.txt") as file:
                            label_text += file.read() + " "
                else:
                    if person in aliases_reversed:
                        possible_aliases = aliases_reversed[person]
                        for alias in possible_aliases:
                            if os.path.isfile(f"output/GPT/character_summaries/{i}/{alias}.txt"):
                                print(f"Searching for chapter {i}, person {person}: found alias {alias}")
                                with open(f"output/GPT/character_summaries/{i}/{alias}.txt") as file:
                                    label_text += file.read() + " "
                                continue
                    
            label_text = '<br>'.join(textwrap.wrap(label_text, width=30)).strip()
            #start_chapter = chapters["Chapter"].min()
            #end_chapter = chapters["Chapter"].max()
           # label_text += f", start chapter: {start_chapter}, end chapter: {end_chapter}"
        # First place the character was introduced so just give the description
        else: 
            text = f"{person}: {character_descriptions[person]}"
            total = '<br>'.join(textwrap.wrap(text, width=30))
            label_text = total

        '''
        if edge[0] in helpers.country_code_to_name:
            label_text += f", start: {helpers.country_code_to_name[edge[0]]}"
        else: 
            label_text += f", start: {edge[0]}"
        if edge[1] in helpers.country_code_to_name:
            label_text += f", end: {helpers.country_code_to_name[edge[1]]}"
        else: 
            label_text += f", end: {edge[1]}"
        '''
        label_data[person].extend([label_text] * 5)


    traces = []

    print(edge_x)

    # Lines between places
    for person in edge_x:
        traces.append(go.Scatter( 
        x=edge_x[person], y=edge_y[person],
        line=dict(width=people_list[person]["width"], dash=people_list[person]["dash"], color=people_list[person]["color"]),
        line_shape='spline',
        #hoverinfo='skip',
        customdata=label_data[person],
        hovertemplate="%{customdata}",
        text=person,
        mode='lines+markers',
        marker=dict(
            symbol="arrow",
            opacity=markers[person],
            angle=90,
            size=20)
        ))

    #print(G)
    character_portraits = os.listdir("pictures/People")

    node_list = []
    for node in people_list:
        #print(node)
        data = {}
        x, y = pos[node]["x"], pos[node]["y"]
        character = node.split("_")[0]
        
        desc = character_descriptions[character]
        total = f"{character}: {desc}"
        total = '<br>'.join(textwrap.wrap(total, width=30))
        data["label"] = character
        data["info"] = "<b>Test something here</b><br>" + total + f"color: {people_list[character]['color']}"
        data["x"] = x
        data["y"] = y
        data["shape"] = G.nodes.data()[node]["shape"]
        data["color"] = G.nodes.data()[node].get("color", "lightblue")
        data["opacity"] = 1.0
        if f"{data['label']}_face.png" in character_portraits:
            data["opacity"] = 0.0
        node_list.append(data)

    df = pd.DataFrame(node_list)

    #print(df)

    #print(df["label"])

    # Character points
    traces.append(go.Scatter(
        x=df["x"], y=df["y"],
        mode='markers+text',
        text=df["label"],
        textposition="top center",
        customdata=df["info"],
        hovertemplate='%{customdata}',
        marker=dict(size=30,symbol=df["shape"],color=df["color"],opacity=df["opacity"])
        ))
              
    fig = go.Figure(data=traces,
                layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    xaxis=dict(showgrid=True, zeroline=False, showticklabels=True),
                    yaxis=dict(showgrid=True, zeroline=False, showticklabels=True, scaleanchor="x", scaleratio=1)
                    )
                )
    print(location_shapes)
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
        country_image = Image.open(f"pictures/Chapters/{path}/{country}.webp")
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
    image = Image.open(f"pictures/background.jpg")

    # Add background image
    fig.add_layout_image(
        x=max_x/2,
        y=max_y/2,
        xanchor="center",
        yanchor="middle",
        source=image,
        xref="x",
        yref="y",
        sizex=max_x * 2,
        sizey=max_y * 2,
        sizing="stretch",
        layer="below",
        opacity=0.5
    )

    for node in node_list:
        character = node["label"]
        #character_portraits = os.listdir("pictures/People")
        if f"{character}_face.png" in character_portraits:
            portrait = Image.open(f"pictures/People/{character}_face.png")
            character_data = node
            print(character_data)
            portrait_size = 384
            print(character, character_data["label"], character_data["x"], character_data["y"])
            #print(f"Adding image for {character} at location: {character_data['label']}, {character_data['y']},")
            fig.add_layout_image(
                        x=character_data["x"],
                        y=character_data["y"],
                        source=portrait,
                        xref="x",
                        yref="y",
                        sizex=portrait_size,
                        sizey=portrait_size,
                        xanchor="center",
                        yanchor="middle",
                        layer="above"
                    )

    #fig.update_yaxes(title='y', visible=False, showticklabels=False, scaleanchor="x", scaleratio=1)
    #fig.update_xaxes(title='x', visible=False, showticklabels=False)
    
    return fig, location_shapes, aspect_ratio, max_x

def add_images(fig, location_shapes, aspect_ratio):
    #img_size_x = 128
    img_size_y = img_size_x# / aspect_ratio
    landscape_aspect_ratio = 7 / 4
    padding = 5
    images_map = {}
    date_df = pd.read_csv("output/GPT/chapter_durations_fixed.csv", sep=";")
    date_df["Start Date"] = pd.to_datetime(date_df["Start Date"])
    date_df["End Date"] = pd.to_datetime(date_df["End Date"])
    start = date_df["Start Date"].min()
    end = start + timedelta(days=80)
    with open("output/chapter_names.txt", encoding="utf-8") as file:
        chapter_names = file.readlines()
    
    for loc in location_shapes:
        #if "test" in location_shapes[loc]:
        #path = location_shapes[loc]['image']
        path = loc
        images = [s for s in os.listdir(f"pictures/Chapters/{path}/") if s.endswith('.webp')]
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
                if f"{filename}.webp" in images:
                    image = Image.open(f"pictures/Chapters/{path}/{filename}.webp")
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
                        summary = '\n'.join(textwrap.wrap(summary, width=60)).strip(),
                    country = loc.split("_")[0]
                    if country in helpers.country_code_to_name:
                        country = helpers.country_code_to_name[country]
                    img_aspect_ratio = image.width / image.height
                    row = date_df.loc[date_df['Chapter'] == int(chapter)].iloc[0]
                    fig.add_trace(
                        go.Scatter(
                            x=[x0,x0,x1,x1,x0], #x1-x1-x2-x2-x1
                            y=[y0,y1,y1,y0,y0], #y1-y2-y2-y1-y1
                            fill="toself",
                            mode='lines',
                            name='',
                            customdata=[{
                                "Image": filename,
                                "Graph": 0,
                                "Country": country,
                                "Chapter": chapter,
                                "Summary": summary,
                                "Aspect Ratio": img_aspect_ratio,
                                "Image Path": f"../../pictures/Chapters/{path}/{filename}.webp",
                                "Chapter Name": '\n'.join(textwrap.wrap(chapter_names[int(chapter)-1], width=50)).strip(),
                                "Start Date": row["Start Date"],
                                "End Date": row["End Date"],
                                "Total Start": start,
                                "Total End": end,
                                }], # Needed information: Chapter name, Image, day
                            hoverinfo="none",
                            #text=f"{country}, Chapter {chapter}: {summary}",
                            text=f"{loc}, Chapter {chapter}, x0: {x0}, x1: {x1}, y0: {y0}, y1: {y1}",
                            opacity=1
                        ))
                    images_map[filename] = image
                else:
                    print(f"WARN: Image name {filename} not found, ignoring it")
            previous_y -= largest_y_in_row

    return fig, images_map


def add_overall_images(fig, location_shapes, aspect_ratio):
    #img_size_x = 128
    #img_size_y = img_size_x * aspect_ratio
    #landscape_aspect_ratio = 7 / 4
    #padding = 5
    date_df = pd.read_csv("output/GPT/chapter_durations_fixed.csv", sep=";")
    date_df["Start Date"] = pd.to_datetime(date_df["Start Date"])
    date_df["End Date"] = pd.to_datetime(date_df["End Date"])
    start = date_df["Start Date"].min()
    end = start + timedelta(days=80)
    images_map = {}
    for loc in location_shapes:
        #if "test" in location_shapes[loc]:
        #path = location_shapes[loc]['image']
        path = loc
        images = [s for s in os.listdir(f"pictures/Chapters/{path}/") if s.endswith('.webp')]
        layout = ""
        with open(f"pictures/Chapters/{loc}/layout_overview", encoding="utf-8") as file:
            layout = file.readline()
        print(images)
        for idx, row in date_df.iterrows():
            countries = row["Country"].split(",")
            date_df.loc[idx,'Include'] = loc in countries
        rows = date_df[date_df["Include"] == True]
        #print(rows)
        #row = -1
        #previous_y = location_shapes[loc]['y1'] - (padding)
        #print(layout)
        # Go through layout file, add the required images
        filename = layout
        if f"{filename}.webp" in images:
            image = Image.open(f"pictures/Chapters/{path}/{filename}.webp")
            img_aspect_ratio = image.width / image.height
            x0, x1 = location_shapes[loc]['x0'], location_shapes[loc]['x1']
            center = (x1 + x0) / 2
            y0, y1 = location_shapes[loc]['y0'], location_shapes[loc]['y1']
            middle = (y0 + y1) / 2
            img_size_x = x1 - x0
            img_size_y = y1 - y0
            summary = ""
            with open(f"output/GPT/summary/{loc}.txt", encoding="utf-8") as file:
                summary = file.read()
                summary = '\n'.join(textwrap.wrap(summary, width=60)).strip(),
            fig.add_trace(
            go.Scatter(
                x=[location_shapes[loc]['x0'],location_shapes[loc]['x0'],location_shapes[loc]['x1'],location_shapes[loc]['x1'],location_shapes[loc]['x0']], #x1-x1-x2-x2-x1
                y=[location_shapes[loc]['y0'],location_shapes[loc]['y1'],location_shapes[loc]['y1'],location_shapes[loc]['y0'],location_shapes[loc]['y0']], #y1-y2-y2-y1-y1
                fill="toself",
                mode='lines',
                name='',
                #hoverinfo="none",
                hoverinfo="text",
                customdata=[{"Image": loc,
                             "Graph": 1,
                             "Start Date": rows["Start Date"].min(),
                             "End Date": rows["End Date"].max(),
                             "Start Chapter": rows['Chapter'].min(),
                             "End Chapter": rows['Chapter'].max(),
                             "Total Start": start,
                             "Total End": end,
                             "Image Path": f"../../pictures/Chapters/{path}/{layout}.webp",
                             "Country": helpers.country_code_to_name[loc.split("_")[0]],
                             "Summary": summary,
                             "Aspect Ratio": img_aspect_ratio}
                             ],
                text=f"{loc}, x0: {location_shapes[loc]['x0']}, x1: {location_shapes[loc]['x1']}, y0: {location_shapes[loc]['y0']}, y1: {location_shapes[loc]['y1']}",
                opacity=1
            ))
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
            images_map[loc] = image
        else:
            print(f"WARN: Image name {filename} not found, ignoring it")

    
    
    return fig, images_map