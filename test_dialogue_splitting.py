import os
import re
from dash import Dash, dcc, html, Input, Output,callback
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import textwrap
import plotly.graph_objects as go



#files = os.listdir("input/Chapters")
files = ["17.txt"]

key_parts = []
figs = []

for file in files:
    with open(f"input/Chapters/{file}", encoding="utf-8") as f:
        text = f.read()
    paragraphs = text.split("\n\n")
    paragraphs = [s.strip().replace("\n", " ") for s in paragraphs]
    paragraphs = list(filter(None, paragraphs))
    print(paragraphs)
    current_text = ""
    for paragraph in paragraphs:
        print(paragraph)
        dialogue = re.findall('\“(.+?)\”', paragraph)
        print("Dialogue:", dialogue)
        dialogue_len = 0
        for line in dialogue:
            dialogue_len += len(line)
        dialogue_percent = dialogue_len / len(paragraph.replace("“", "").replace("”", ""))
        print("Dialogue percentage:", dialogue_percent)
        if dialogue_percent <= 0.1:
            current_text += " " + paragraph
        else:
            if current_text != "":
                key_parts.append(current_text)
                current_text = ""

    for par in key_parts:
        print(par, "\n")

with open("output/GPT/summary/17_paragraph.txt", encoding="utf-8") as f:
    summary_sentences = f.readlines()
    
'''
#print(key_parts)
own_model = SentenceTransformer("dump/en_simcse_80days/")
pretrained_model = SentenceTransformer("all-mpnet-base-v2")

pretrained_embeddings = pretrained_model.encode(key_parts, normalize_embeddings=True)
own_embeddings = own_model.encode(key_parts, normalize_embeddings=True)

pretrained_embeddings_summary = pretrained_model.encode(summary_sentences, normalize_embeddings=True)
own_embeddings_summary = own_model.encode(summary_sentences, normalize_embeddings=True)

#print(embeddings)
pretrained_similarities = []
own_similarities = []


#example_sentence = summary_sentences[test]
#print("Example sentence:", example_sentence)
for i in range(0, len(summary_sentences)):
    pretrained_similarities = []
    own_similarities = []
    x = []
    text = []
    for j in range(0, len(key_parts)):
        diff = util.dot_score(pretrained_embeddings_summary[i], pretrained_embeddings[j]).item()
        #print(diff)
        pretrained_similarities.append(diff)
        own_diff = util.dot_score(own_embeddings_summary[i], own_embeddings[j]).item()
        #print(diff)
        own_similarities.append(own_diff)
        x.append(j)
        text.append('<br>'.join(textwrap.wrap(summary_sentences[i] + "<br>/<br>" + key_parts[j], width=60)).strip(),)


    fig = go.Figure()

    fig.add_trace(go.Line(x=x, y=pretrained_similarities, text=text, name="Pretrained model"))
    fig.add_trace(go.Line(x=x, y=own_similarities, text=text, name="Own model"))

    fig.update_traces(mode='lines+markers')
    figs.append(fig)
    
for fig in figs:
    fig.show()
'''

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
'''
example_sentence = summary_sentences[test]
print("Example sentence:", example_sentence)

for i in range(0, len(key_parts) - 1):
    diff = util.dot_score(pretrained_embeddings[i], pretrained_embeddings_summary[test]).item()
    #print(diff)
    pretrained_similarities.append(diff)
    own_diff = util.dot_score(own_embeddings[i], own_embeddings_summary[test]).item()
    #print(diff)
    own_similarities.append(own_diff)
    x.append(i)
    text.append('<br>'.join(textwrap.wrap(example_sentence + "<br>/<br>" + key_parts[i], width=60)).strip(),)


fig = go.Figure()

fig.add_trace(go.Line(x=x, y=pretrained_similarities, text=text, name="Pretrained model"))
fig.add_trace(go.Line(x=x, y=own_similarities, text=text, name="Own model"))

fig.update_traces(mode='lines+markers')
fig.show()

'''