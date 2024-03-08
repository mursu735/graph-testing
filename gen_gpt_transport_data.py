import os
import re
import openai
from datetime import datetime
import time
import helpers
import tiktoken

def determine(file):
    filename = file.split(".")[0]
    return os.path.isfile(f"output/GPT/transport/{filename}")

files = os.listdir("input/Chapters")
files = helpers.natural_sort([file for file in files if not determine(file)])
print(f"Running for files {files}")

key = ""
with open("key.txt") as file:
    key = file.read()
print(key)

openai.api_key = key

# list models
#models = openai.Model.list()

# print the first model's id
#print(models)

with open("input/instruction_transport.txt", encoding="utf-8") as f:
    instruction = f.read()

start_time = datetime.now()

for file in files:
    chapter = file.split(".")[0]
    with open(f"input/Chapters/{file}", encoding="utf-8") as f:
        text = "prompt:\n" + f.read()

    whole_prompt = instruction + text

    with open(f"dump/transport_{chapter}.txt", "w", encoding="utf-8") as file:
        file.write(whole_prompt)

    # Check for number of tokens sent in the last minute
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(whole_prompt))
    print(num_tokens)

    # create a chat completion
    print("Creating GPT request")
    current_start = datetime.now()
    chat_completion = openai.ChatCompletion.create(model="gpt-4-0125-preview", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": text}], temperature=0)
    #chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}])
    # print the chat completion
    message = chat_completion.choices[0].message.content
    print(message)
    prompt_tokens = chat_completion.usage["prompt_tokens"]
    completion_tokens = chat_completion.usage["completion_tokens"]
    total_tokens = chat_completion.usage["total_tokens"]

    print("Total tokens used", total_tokens)
    current_time = datetime.now()
    difference = (current_time - current_start).total_seconds()

    print("Time taken:", difference, "seconds")
    '''
    if not os.path.isfile(f"output/GPT/tokens/tokens_transport.csv"):
        with open("output/GPT/tokens/tokens_paragraph_summary.csv", "w", encoding='utf-8') as file:
            file.write("Chapter;Prompt;Completion;Total;Time\n")

    with open("output/GPT/tokens/tokens_paragraph_summary.csv", "a", encoding="utf-8") as file:
        file.write(f"{chapter};{prompt_tokens};{completion_tokens};{total_tokens};{difference}\n")
    print(message)
    '''
    with open(f"output/GPT/transport/{chapter}", "w", encoding="utf-8") as file:
        file.write(message)


current_time = datetime.now()
difference = (current_time - start_time).total_seconds()

print("Time taken for all prompts:", difference, "seconds")