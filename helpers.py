import re

def get_instruction():
    instruction = ""
    with open("input/instruction_location.txt") as file:
        instruction = ''.join(line for line in file)
    return instruction

def get_summary_instruction():
    instruction = ""
    with open("input/instruction_character_summary.txt") as file:
        instruction = ''.join(line for line in file)
    return instruction

def get_country_summary_instruction():
    instruction = ""
    with open("input/instruction_country_summary.txt") as file:
        instruction = ''.join(line for line in file)
    return instruction

def get_chapter(chapter, add_prompt=True):
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
        if add_prompt:
            prompt += f"prompt:{text}"
        else:
            prompt += text
    return prompt
    
def build_prompt(chapter):
    prompt = get_instruction() + "\n" + get_chapter(chapter)
    return prompt

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_character_descriptions():
    descriptions = {}
    with open("output/character_descriptions.txt", encoding="utf-8") as file:
        text = file.readlines()

    for line in text:
        if line != "\n":
            line = line.replace("\n", "")
            #print(line)
            character, description = line.split(":")
            descriptions[character] = description
    return descriptions

def get_aliases():
    return {"Mr. Phileas Fogg": "Phileas Fogg", "Mr. Fogg": "Phileas Fogg", "Jean Passepartout": "Passepartout", "Detective Fix": "Fix", "John Busby": "John Bunsby", "Colonel Proctor": "Colonel Stamp Proctor", "FIX, Detective": "Fix",
            "British consul at Suez": "British consul", "Consul": "British consul", "Detective (Mr. Fix)": "Fix", "The consul": "British consul", "Brigadier-general of the English army": "Sir Francis Cromarty", "The Parsee": "Parsee",
            "The guide": "Parsee", "Detective (Fix)": "Fix", "Mr. Batulcar": "Honourable William Batulcar", "A belated Mormon": "Mormon Man", "Forster (the engineer)": "James Forster", "Captain Speedy": "Andrew Speedy"}

def get_aliases_reversed():
    aliases = get_aliases()
    aliases_reverse = {}
    for alias in aliases:
        #print(alias)
        actual_name = aliases[alias]
        if actual_name not in aliases_reverse:
            aliases_reverse[actual_name] = []
        aliases_reverse[actual_name].append(alias)
    return aliases_reverse


country_code_to_name = {"gb": "Great Britain", "fr": "France", "eg": "Egypt", "ye": "Yemen", "in": "India", "sg": "Singapore", "my": "Malaysia", "hk": "Hong Kong", "cn": "China", "jp": "Japan", "us": "USA", "ie": "Ireland", "sea": "At sea"}

def get_clusters():
    return {"Reverend Samuel Wilson": 1, "James Forster": 2, "Sir Francis Cromarty": 3, "Elder William Hitch": 4, "Mormon Man": 4, "Lord Albemarle": 5, "British consul": 5, "Fix": 6, "Aouda": 6, "Phileas Fogg": 6, "Passepartout": 6,
            "John Bunsby": 7, "Honourable William Batulcar": 7, "Colonel Stamp Proctor": 8, "Mudge": 8, "Andrew Speedy": 8, "Parsee": 9, "James Strand": 10, "Gauthier Ralph": 11, "Thomas Flanagan": 11, "Samuel Fallentin": 11,
            "Andrew Stuart": 11, "John Sullivan": 11}
    # Old clustering
    '''
    return {"Sir Francis Cromarty": 1, "James Forster": 2, "Phileas Fogg": 3, "Passepartout": 3, "Andrew Stuart": 3, "Andrew Stuart_2": 3, "British consul": 4, 
            "Reverend Samuel Wilson": 5, "John Bunsby": 5, "Aouda": 6, "Parsee": 6, "Samuel Fallentin": 7, "Samuel Fallentin_2": 7, "John Sullivan": 7, "John Sullivan_2": 7, "Gauthier Ralph": 7, "Gauthier Ralph_2": 7, 
            "Thomas Flanagan": 7, "Thomas Flanagan_2": 7, "Lord Albemarle": 7, "Fix": 7, "Elder William Hitch": 8, "Honourable William Batulcar": 8, "Mormon Man": 8,  "Colonel Stamp Proctor": 8,
            "Andrew Speedy": 9, "Mudge": 9, "James Strand": 10}
    '''
