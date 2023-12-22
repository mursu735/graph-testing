import re

def get_instruction():
    instruction = ""
    with open("input/instruction_location.txt") as file:
        instruction = ''.join(line for line in file)
    return instruction

def get_chapter(chapter):
    prompt = ""
    '''
    if chapter > 1:
        with open(f"output/GPT/summary/{chapter - 1}.txt", encoding="utf-8") as file:
            text = file.read()
            prompt += f"context:\n{text}"
    '''
    prompt += "\n\n"
    with open(f"input/Chapters/{chapter}.txt", encoding="utf-8") as file:
        text = file.read()
        prompt += f"prompt:\n{text}"
    return prompt
    
def build_prompt(chapter):
    prompt = get_instruction() + "\n" + get_chapter(chapter)
    return prompt

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_aliases():
    return {"Mr. Phileas Fogg": "Phileas Fogg", "Mr. Fogg": "Phileas Fogg", "Jean Passepartout": "Passepartout", "Detective Fix": "Fix", "John Busby": "John Bunsby", "Colonel Proctor": "Colonel Stamp Proctor"}
