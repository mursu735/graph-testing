import spacy
import en_core_web_trf
import en_core_web_sm
import re
import os
import pandas as pd

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

#df = pd.read_csv("output/Chapters/1.csv", sep=";", names=["name", "loc", "pos"])
#for idx, row in df.iterrows():
#    print(row["name"])

files = os.listdir("output/Chapters")
for file in files: 
    name = file.split(".")[0]
    #name = split[0]
    print(name)
# display DataFrame
#print(df["Location"])
#line = '["Bildad", "hires", "Ishmael"]'
#line = line.replace("[", "").replace("]", "").replace('"', '').replace("\n", "")
#doc = nlp(line)
#print([(w.text, w.pos_) for w in doc])

#for ent in doc.ents:
#    print(ent.text, ent.label_)