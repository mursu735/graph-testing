import os
import helpers

summaries = helpers.natural_sort(os.listdir("output/GPT/summary"))

print(summaries)

text = ""

for filename in summaries:
    text += "Chapter " + filename.split(".")[0] + ":\n"
    with open(f"output/GPT/summary/{filename}", encoding="utf-8") as file:
        text += "n".join(file.readlines()).replace(". ", ".\n")
    if filename is not summaries[-1]:
        text += "\n\n"

with open("complete_summary.txt", "w", encoding="utf-8") as file:
    file.write(text) 