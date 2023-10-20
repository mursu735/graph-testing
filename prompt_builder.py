import os


chapter = int(input("Enter chapter (1-37):"))

prompt = ""

with open("input/instruction_location.txt", encoding="utf-8") as file:
    prompt += file.read()

prompt = prompt.replace("<chapter>", str(chapter))

prompt += "\n"
if chapter > 1:
    with open(f"output/GPT/summary/{chapter - 1}.txt", encoding="utf-8") as file:
        text = file.read()
        prompt += f"context:\n{text}"

prompt += "\n"
with open(f"input/Chapters/{chapter}.txt", encoding="utf-8") as file:
    text = file.read()
    prompt += f"prompt:\n{text}"

#print(prompt)

with open("prompt.txt", "w", encoding="utf-8") as file:
    file.write(prompt)

print("Prompt has been written to prompt.txt")