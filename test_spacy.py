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

df = pd.read_csv("output/GPT/locations.csv", sep=";")
locations = df.groupby("Chapter")
print(locations.get_group(7).count()["Location"])
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