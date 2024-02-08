import string

titles = []

with open("input/around the world.txt", encoding="utf-8") as file:
    
    text = file.readlines()
    for i in range(0, len(text)):
        current = ""
        line = text[i]
        # Found a chapter, go over the text until whitespace is found
        if "CHAPTER" in line:
            while line != "\n":
                i+=1
                line = text[i]
                current += line
            current = string.capwords(current.replace("\n", " ").strip())
            titles.append(current)

print(titles)

with open("output/chapter_names.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(titles))