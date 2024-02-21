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
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import helpers
from dash import Dash, dcc, html, Input, Output,callback


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


files = helpers.natural_sort(os.listdir("input/Chapters"))
#files = ["11.txt"]
figs = []

if not os.path.isdir("output/GPT/similarities"):
    os.mkdir("output/GPT/similarities")

print(files)
for file in files:
    sentences = []
    with open(f"input/Chapters/{file}", encoding="utf-8") as f:
        text = f.read()
    paragraphs = text.split("\n\n")
    paragraphs = [s.strip().replace("\n", " ") for s in paragraphs]
    paragraphs = list(filter(None, paragraphs))
    for paragraph in paragraphs:
        sentences.append(paragraph)

    print(sentences)
    own_model = SentenceTransformer("dump/en_simcse_80days/")
    pretrained_model = SentenceTransformer("all-mpnet-base-v2")

    pretrained_embeddings = pretrained_model.encode(sentences, normalize_embeddings=True)
    own_embeddings = own_model.encode(sentences, normalize_embeddings=True)

    #print(embeddings)
    pretrained_similarities = []
    own_similarities = []
    x = []
    text = []

    for i in range(0, len(sentences) - 1):
        diff = util.dot_score(pretrained_embeddings[i], pretrained_embeddings[i+1]).item()
        #print(diff)
        pretrained_similarities.append(diff)
        own_diff = util.dot_score(own_embeddings[i], own_embeddings[i+1]).item()
        #print(diff)
        own_similarities.append(own_diff)
        x.append(i)
        text.append('<br>'.join(textwrap.wrap(sentences[i] + "/" + sentences[i+1], width=60)).strip(),)


    fig = go.Figure()

    fig.add_trace(go.Line(x=x, y=pretrained_similarities, text=text, name="Pretrained model"))
    fig.add_trace(go.Line(x=x, y=own_similarities, text=text, name="Own model"))

    fig.update_traces(mode='lines+markers')
    as_text = fig.to_json()
    filename = file.split(".")[0]
    with open(f"output/GPT/similarities/{filename}.json", "w") as file:
        file.write(as_text)


#fig.show()

#model_output_path = f'dump/en_simsce_80days-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

'''
app = Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(list(range(1, len(figs)+1)), 1, id='demo-dropdown'),
    dcc.Graph(figure=figs[0], id='dropdown-graph')
])


@callback(
    Output('dropdown-graph', 'figure'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    return figs[value-1]


if __name__ == '__main__':
    app.run(debug=True)
'''