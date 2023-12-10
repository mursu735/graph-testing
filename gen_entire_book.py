import os
import helpers
import gen_pygraphviz
import pandas as pd


prefix = "output/GPT/locations"
files = helpers.natural_sort(os.listdir(prefix))

print(files)


if not os.path.isfile("whole_book.csv"):
    df = pd.DataFrame()
    for file in files:
        chapter = pd.read_csv(f"{prefix}/{file}", sep=";")
        chapter_number = file.split(".")[0]
        chapter["Chapter"] = chapter_number
        df = pd.concat([df, chapter], ignore_index=True)

    df.to_csv("whole_book.csv", sep=";")

path = "whole_book.csv"

fig = gen_pygraphviz.generate_graph(path)

fig.show()