import os
import pandas as pd

files = os.listdir("output/GPT/transport")


df = pd.DataFrame()

for file in files:
    with open(f"output/GPT/transport/{file}", encoding="utf-8") as f:
        methods = f.readlines()
    methods = [method.replace("-", "").strip() for method in methods]
    chapter = pd.DataFrame({'Transport': methods, 'Chapter': file})
    df = pd.concat([df, chapter], ignore_index=True)

df.to_csv('output/GPT/transport.csv', sep=";", index=False)  

transport = pd.read_csv("output/GPT/transport.csv", sep=";")
print(list(transport.columns))


for chapter in files:
    print(transport.loc[transport["Chapter"] == int(chapter)]["Transport"].tolist())
