# The code below is a modified version of the forllowing file.
# sentence-transformers: examples/unsupervised_learning/SimCSE/train_simcse_from_file.py

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import logging
import plotly
import textwrap
import plotly.express as px
import spacy
from datetime import datetime
import sys
import os
import helpers



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
sentences = []

#files = helpers.natural_sort(os.listdir("input/Chapters"))
files = ["11.txt"]

nlp = spacy.load('en_core_web_sm')

print(files)
for file in files:
    with open(f"input/Chapters/{file}", encoding="utf-8") as file:
        text = file.read()
    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        text = nlp(paragraph).sents
        current_paragraph = []
        for sentence in text:
            current_paragraph.append(sentence.text.strip().replace("\n", " "))
        sentences.append(current_paragraph)
sentences = [item for item in sentences if item != [] and item != ['']]
print(sentences)

model = SentenceTransformer("dump/en_simcse_80days/")

paragraph_embeddings = []

for paragraph in sentences:

    embeddings = model.encode(paragraph, normalize_embeddings=True)
    paragraph_embeddings.append(embeddings)
    #print(embeddings)

similarities = []
x = []
text = []

for i in range(0, len(paragraph_embeddings) - 1):
    diff = util.dot_score(paragraph_embeddings[i], paragraph_embeddings[i+1])
    print(diff)
    '''
    similarities.append(diff)
    x.append(i)
    text.append('<br>'.join(textwrap.wrap(" ".join(sentences[i]) + "/" + " ".join(sentences[i+1]), width=60)).strip(),)


fig = px.line(x=x, y=similarities, text=text)

fig.update_traces(mode='lines+markers')

fig.show()'''