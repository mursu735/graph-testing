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
files = ["27.txt"]

print(files)
for file in files:
    with open(f"input/Chapters/{file}", encoding="utf-8") as file:
        text = file.read()
    paragraphs = text.split("\n\n")
    paragraphs = [s.strip().replace("\n", " ") for s in paragraphs]
    paragraphs = list(filter(None, paragraphs))
    for paragraph in paragraphs:
        sentences.append(paragraph)

print(sentences)
model = SentenceTransformer("dump/en_simcse_80days/")

embeddings = model.encode(sentences, normalize_embeddings=True)

print(embeddings)
similarities = []
x = []
text = []

for i in range(0, len(sentences) - 1):
    diff = util.dot_score(embeddings[i], embeddings[i+1]).item()
    print(diff)
    similarities.append(diff)
    x.append(i)
    text.append('<br>'.join(textwrap.wrap(sentences[i] + "/" + sentences[i+1], width=60)).strip(),)


fig = px.line(x=x, y=similarities, text=text)

fig.update_traces(mode='lines+markers')

fig.show()

#model_output_path = f'dump/en_simsce_80days-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
