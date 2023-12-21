import os

if not os.path.isdir("./input/Chapters"):
    os.mkdir("./input/Chapters")

with open("input/around the world.txt", encoding="utf-8") as file:
    lines = file.readlines()
    file_name = 0
    text = ""
    for line in lines:
        if "CHAPTER" in line:
            if file_name > 0:
                with open(f"input/Chapters/{file_name}.txt", "w", encoding="utf-8") as f:
                    f.write(text)
                    text = ""
            file_name += 1
        if not line.isupper():
            text += line
    with open(f"input/Chapters/{file_name}.txt", "w", encoding="utf-8") as f:
        f.write(text)