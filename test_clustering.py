import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
import helpers


def create_heatmap(matrix, character_list):
    fig2 = ff.create_dendrogram(matrix, labels=character_list)
    for i in range(len(fig2['data'])):
        fig2['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(matrix, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig2.add_trace(data)

    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))

    data_dist = pdist(matrix)
    heat_data = squareform(data_dist)
    heat_data = heat_data[dendro_leaves,:]
    heat_data = heat_data[:,dendro_leaves]

    print("Heatmap data:\n", heat_data)

    heatmap = [
        go.Heatmap(
            x = dendro_leaves,
            y = dendro_leaves,
            z = heat_data,
            colorscale = 'RdBu'
        )
    ]

    heatmap[0]['x'] = fig2['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig2.add_trace(data)

    fig2['layout']['yaxis']['ticktext'] = fig2['layout']['xaxis']['ticktext']
    fig2['layout']['yaxis']['tickvals'] = np.asarray(dendro_side['layout']['yaxis']['tickvals'])

    # Edit Layout
    fig2.update_layout({'width':800, 'height':800,
                            'showlegend':False, 'hovermode': 'closest',
                            })
    # Edit xaxis
    fig2.update_layout(xaxis={'domain': [.15, 1],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'ticks':""})
    # Edit xaxis2
    fig2.update_layout(xaxis2={'domain': [0, .15],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks':""})

    # Edit yaxis
    fig2.update_layout(yaxis={'domain': [0, .85],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks': ""
                            })
    # Edit yaxis2
    fig2.update_layout(yaxis2={'domain':[.825, .975],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks':""})

    #fig2.update_layout(title = text)

    fig2.show()

    



df = pd.read_csv("whole_book.csv", sep=";")
aliases = helpers.get_aliases()

# 1. Go through all chapters, list people in them
# 2. Go through all lists and combinations, add one to all combinations

character_appearances = {}
character_list = []
for idx, row in df.iterrows():
    #print(row) 
    characters = row["Person"].split("|")
    #print(characters)
    for character in characters:
        if character in aliases:
            character = aliases[character]
        chapter = row["Chapter"]
        if character not in character_list:
            character_list.append(character)
        if chapter not in character_appearances:
            character_appearances[chapter] = []
        if character not in character_appearances[chapter]:
            character_appearances[chapter].append(character)
        
print(character_appearances)
print(character_list)
last_chapter = df["Chapter"].max()

appearances_together = np.zeros((len(character_list), len(character_list)))
print(appearances_together)

for chapter in character_appearances:
    characters = character_appearances[chapter]
    #print(characters)
    for i in range(0, len(characters)):
        for j in range(0, len(characters)):
            first_character = character_list.index(characters[i])
            second_character = character_list.index(characters[j])
            #print(characters[i], first_character, characters[j], second_character)
            appearances_together[first_character, second_character] += 1
            #appearances_together[j, i] += 1

print(appearances_together)


#appearance = np.zeros((len(character_list), last_chapter))


'''
for idx, row in df.iterrows():
    #print(row) 
    characters = row["Person"].split("|")
    #print(characters)
    for character in characters:
        if character in aliases:
            character = aliases[character]
        chapter = row["Chapter"]
        index = character_list.index(character)
        appearance[index][chapter-1] = 1

for character in character_list:
    print(character)
    index = character_list.index(character)
    print(appearance[index])
'''
distances = np.zeros((len(character_list), len(character_list)))

for i in range(0, len(character_list)):
    current = appearances_together[:, i]
    for j in range(0, len(character_list)):
        comparison = appearances_together[:, j]
        cosine = np.dot(current, comparison) / (np.linalg.norm(current) * np.linalg.norm(comparison))
        distances[i, j] = cosine


similarity_of_similarities = np.zeros((len(character_list), len(character_list)))

for i in range(0, len(character_list)):
    current = distances[:, i]
    for j in range(0, len(character_list)):
        comparison = distances[:, j]
        cosine = np.dot(current, comparison) / (np.linalg.norm(current) * np.linalg.norm(comparison))
        similarity_of_similarities[i, j] = cosine


#create_heatmap(similarity_of_similarities, character_list)
heat_data, order, character_list = create_heatmap(distances, character_list)

fogg = character_list.index("Phileas Fogg")

print(order)
