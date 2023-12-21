import os
import gen_pygraphviz
import networkx as nx
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import random
from geopy.geocoders import Nominatim
import helpers



path = "output/GPT/locations/17.csv"

fig = gen_pygraphviz.generate_graph(path)

fig.show()

'''
for index, row in df.iterrows(): 
    # Create list for the current row 
    lat = row.Latitude
    lon = row.Longitude
    coord = str(lat) + "," + str(lon)
    if coord not in visited_locations:
        location = geolocator.reverse(coord)
        print(location)
        if location is not None:
            row["Country"] = location.raw['address']['country_code']
            visited_locations[coord] = location.raw['address']['country_code']
        else:
            row["Country"] = "sea"
            visited_locations[coord] = "sea"
    else:
        row["Country"] = visited_locations[coord]
    # Display
    print(location)
   ''' 

'''
locations = []
#for lat, lon in locations:
for i in range(175, 180):
    lat, lon = locations[i]
    location = geolocator.reverse(str(lat)+","+str(lon))
 
    # Display
    print(location)
    #print(location.raw['address']['country_code'])
    if location is not None:
        print(location.raw['address']['country_code'])
    else:
        print("sea")

'''
'''
G = nx.MultiDiGraph()

df = pd.read_csv(f"{path}", sep=";")

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
            G.add_node(person, shape="circle", c=col)
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

#G.layout()

#print(people_ending_in_location)

#pos = nx.spring_layout(G)
pos = nx.drawing.nx_agraph.pygraphviz_layout(
    G,
    prog='dot',
    args='-Grankdir=LR' + ' ' + '-Gnewrank=true' + ' ' + '-Gsplines=true'
)

nx.draw_networkx(G, pos=pos, with_labels = True)
plt.show()
'''