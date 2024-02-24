import os
import re
from dash import Dash, dcc, html, Input, Output,callback
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import textwrap
import openai
from datetime import datetime
import time
import tiktoken
import plotly.graph_objects as go

def determine(file):
    filename = file.split(".")[0]
    return os.path.isfile(f"output/GPT/summary/{filename}_paragraph.txt")

files = os.listdir("input/Chapters")
files = [file for file in files if not determine(file)]
print(f"Running for files {files}")
#iles = ["17.txt"]


figs = []

key = ""
with open("key.txt") as file:
    key = file.read()
print(key)

openai.api_key = key

# list models
#models = openai.Model.list()

# print the first model's id
#print(models)

with open("input/instruction_paragraph_summary.txt", encoding="utf-8") as f:
    instruction = f.read()

token_per_min = 60000
total_used = 0
start_time = datetime.now()

for file in files:
    key_parts = []
    with open(f"input/Chapters/{file}", encoding="utf-8") as f:
        text = f.read()
    paragraphs = text.split("\n\n")
    paragraphs = [s.strip().replace("\n", " ") for s in paragraphs]
    paragraphs = list(filter(None, paragraphs))
    #print(paragraphs)
    current_text = ""
    for paragraph in paragraphs:
        #print(paragraph)
        dialogue = re.findall('\“(.+?)\”', paragraph)
        #print("Dialogue:", dialogue)
        dialogue_len = 0
        for line in dialogue:
            dialogue_len += len(line)
        dialogue_percent = dialogue_len / len(paragraph.replace("“", "").replace("”", ""))
        #print("Dialogue percentage:", dialogue_percent)
        if dialogue_percent <= 0.1:
            key_parts.append(paragraph)
       #
        #else:
        #    if current_text != "":
        #        key_parts.append(current_text)
        #        current_text = ""
       # 
    #for par in key_parts:
    #    print(par, "\n")
    filename = file.split(".")[0]

    final_text = "\n\n".join(key_parts)

    '''
    with open(f"dump/paragraphs/{filename}_paragraph.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(key_parts))
    '''


    prompt = "prompt:\n" + final_text 
    #filename = file
    print(f"Processing chapter {file}")
    #number += 1
    whole_prompt = instruction + "\n" + prompt

    with open(f"dump/paragraphs/{filename}_paragraph.txt", "w", encoding="utf-8") as f:
        f.write(whole_prompt)

    # Check for number of tokens sent in the last minute
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    num_tokens = len(encoding.encode(whole_prompt))
    print(num_tokens)
    with open(f"dump/paragraphs/{filename}_paragraph.txt", "w", encoding="utf-8") as f:
        f.write(whole_prompt)
    current_time = datetime.now()
    difference = (current_time - start_time).total_seconds()
    # Only check if this not the first prompt within the last minute
    if difference < 60:
        tokens_used = total_used + num_tokens
        if tokens_used > token_per_min:
            print("Token limit per minute reached, sleep for 60 seconds")
            time.sleep(60)
            total_used = 0
            start_time = datetime.now()
        else:
            print(f"Tokens used since last sleep {tokens_used}/{token_per_min}, no need to sleep")
    else:
        print("Over one minute elapsed since the first prompt, reseting token count")
        total_used = 0
        start_time = datetime.now()
    
    print("Send request to GPT")
    # create a chat completion
    last_request = time.time()
    chat_completion = openai.ChatCompletion.create(model="gpt-4-0125-preview", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}], temperature=0)
    #chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}])
    # print the chat completion
    message = chat_completion.choices[0].message.content
    print(message)
    prompt_tokens = chat_completion.usage["prompt_tokens"]
    completion_tokens = chat_completion.usage["completion_tokens"]
    total_tokens = chat_completion.usage["total_tokens"]
    total_used += total_tokens
    
    #if not os.path.isfile(f"output/GPT/tokens/tokens_paragraph_summary.csv"):
    #    with open("output/GPT/tokens/tokens_paragraph_summary.csv", "w", encoding='utf-8') as file:
    #        file.write("Chapter;Prompt;Completion;Total\n")

    #with open("output/GPT/tokens/tokens_paragraph_summary.csv", "a", encoding="utf-8") as file:
    #    file.write(f"{filename};{prompt_tokens};{completion_tokens};{total_tokens}\n")
    #print(message)
    
    # Put all main locations to single file
    #locations = parts[0]
    with open(f"output/GPT/summary/{filename}_paragraph.txt", "w", encoding="utf-8") as file:
        file.write(message)
