from helpers import build_prompt

prompt = build_prompt(int(input("Enter chapter (1-37):")))
with open("prompt.txt", "w", encoding="utf-8") as file:
    file.write(prompt)

print("Prompt has been written to prompt.txt")