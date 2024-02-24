import os
import helpers

summaries = os.listdir("output/GPT/summary")
summaries = helpers.natural_sort([summary for summary in summaries if "paragraph" in summary])
print(summaries)

with open("input/around the world.txt", encoding="utf-8") as f:
    full_text = f.read()

for summary in summaries:
    chapter = summary.split("_")[0]
    print("Processing chapter", chapter)
    with open(f"output/GPT/summary/{summary}", encoding="utf-8") as file:
        lines = file.readlines()
    lines = [s.strip().replace("\n", " ") for s in lines]
    lines = list(filter(None, lines))
    #print(lines)
    for line in lines:
        whole_text, summary = line.split("--")
        whole_text = whole_text.strip().replace('"', "")
        if whole_text not in full_text:
            print(f"ERROR for chapter {chapter}: text '{whole_text}' not found in full text")
    print("---------------------")