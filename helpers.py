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

country_code_to_name = {"gb": "Great Britain", "fr": "France", "eg": "Egypt", "ye": "Yemen", "in": "India", "sg": "Singapore", "my": "Malaysia", "hk": "Hong Kong", "cn": "China", "jp": "Japan", "us": "USA", "ie": "Ireland"}

def get_clusters():
    return {"Sir Francis Cromarty": 1, "James Forster": 2, "Phileas Fogg": 3, "Passepartout": 3, "Andrew Stuart": 3, "Andrew Stuart_2": 3, "British consul": 4, 
            "Reverend Samuel Wilson": 5, "John Bunsby": 5, "Aouda": 6, "Parsee": 6, "Samuel Fallentin": 7, "Samuel Fallentin_2": 7, "John Sullivan": 7, "John Sullivan_2": 7, "Gauthier Ralph": 7, "Gauthier Ralph_2": 7, 
            "Thomas Flanagan": 7, "Thomas Flanagan_2": 7, "Lord Albemarle": 7, "Fix": 7, "Elder William Hitch": 8, "Honourable William Batulcar": 8, "Mormon Man": 8,  "Colonel Stamp Proctor": 8,
            "Andrew Speedy": 9, "Mudge": 9, "James Strand": 10}
