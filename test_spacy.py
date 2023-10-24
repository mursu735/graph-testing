import spacy
import en_core_web_trf
import en_core_web_sm
import re
import os
import pandas as pd
import numpy as np
import random
import plotly.express as px
from textwrap import wrap

'''
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
#nlp = en_core_web_trf.load()
line = '["Moby Dick", "could\'ve", "Ahab\'s obsession"]'
line = line.replace("[", "").replace("]", "").replace('"', '').replace("\n", "")
my_list = re.split(r" |'", line)
print(my_list)
doc = nlp(line)
print([(w.text.replace("'", ""), w.pos_) for w in doc])
'''
'''
df = pd.read_csv("output/GPT/locations/1.csv", sep=";")
groups = {}
#df["Latitude"] = df["Latitude"] + random.uniform(-0.005, 0.005)
#df2 = pd.DataFrame(np.random.uniform(-0.005,0.005,size=(df.shape[0], 2)), columns=['lat', 'long'])
print(df)
for idx, row in df.iterrows():
    if "|" in row["Person"]:
        if not row["Person"] in groups:
            cur = len(groups) + 1
            groups[row["Person"]] = {"group": f"Group {cur}", "locations": []}
        groups[row["Person"]]["locations"].append(row["Location"])
'''

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

locations = []
directory = "output/GPT/locations"
files = os.listdir(directory)
files = natural_sort(files)
# TODO: Change all of the chapters to the new format and uncomment these
for file in files:
    df = pd.read_csv(f"{directory}/{file}", sep=";")
    df = df.dropna()
    df = df.astype({'Latitude':'float','Longitude':'float'})
    df["Label"] = "Chapter " + file.split(".")[0] + ", " + df["City"]
    df["Chapter"] = int(file.split(".")[0])
    #df["Order"] = df["index"]
    df = df.loc[df['Person'].str.contains("Fogg")]
    locations.append(df)

#print(locations)
df = pd.concat(locations)
#df.sort_values(by=["Chapter"])
#print(df)
df = df.reset_index()
df.to_csv("test.csv")
'''
for file in files:
    df = pd.read_csv(f"{directory}/{file}", sep=";")
    df.dropna()
    df["Chapter"] = "Chapter " + file.split(".")[0] + ", " + df["City"]
    df = df.loc[df['Person'].str.contains("Fogg")]
    print(f"File: {file}", df.dtypes["Latitude"])
    locations.append(df)
'''
#print(locations)
#result = pd.concat(locations)
#print(result)
#print(result.shape[0])
print(df.dtypes)
#result.to_csv("test.csv")
#print(locations.loc[df.loc[df['Chapter'] == "7"]])
#print(named_colorscales[13])
# display DataFrame
#print(df["Location"])
#line = '["Bildad", "hires", "Ishmael"]'
#line = line.replace("[", "").replace("]", "").replace('"', '').replace("\n", "")
#doc = nlp(line)
#print([(w.text, w.pos_) for w in doc])

#for ent in doc.ents:
#    print(ent.text, ent.label_)