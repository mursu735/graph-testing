import os
import re
import time
import openai
import tiktoken
import helpers
import pandas as pd


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def determine(file):
    return os.path.isfile(f"output/GPT/summary/{file}.txt")

key = ""
with open("key.txt") as file:
    key = file.read()
print(key)

openai.api_key = key

# list models
#models = openai.Model.list()

# print the first model's id
#print(models.data)

instruction = helpers.get_country_summary_instruction()
print(instruction)

if not os.path.exists("output/GPT"):
    os.mkdir("output/GPT")
    
#if not os.path.isfile("output/GPT/locations.csv"):
#   with open("output/GPT/locations.csv", "w", encoding="utf-8") as file:
#       file.write("Location;Latitude;Longitude;Order\n")

#if not os.path.exists("output/GPT/character_summaries"):
    
date_df = pd.read_csv("output/GPT/chapter_durations_fixed.csv", sep=";")
country_list = set()
chapters_in_country = {}

for idx, row in date_df.iterrows():
    countries = row["Country"].split(",")
    for country in countries:
        country_list.add(country)

for country in country_list:
    for idx, row in date_df.iterrows():
        countries = row["Country"].split(",")
        date_df.loc[idx,'Include'] = country in countries
    rows = date_df[date_df["Include"] == True]
    #print(country, "\n", rows)
    chapters_in_country[country] = {"start": rows["Chapter"].min(), "end": rows["Chapter"].max()}

countries = list(chapters_in_country.keys())



#files = os.listdir("input/Chapters")
#files = natural_sort(files)
#print("Before:")
#print(files)
# Only include chapters that haven't been parsed yet to reduce load and cost
files = [file for file in countries if not determine(file)]
first = True
print(f"Processing files {files}")
'''
last_request = time.time()

for filename in files:
    prompt = "prompt:"
    #filename = file
    print(f"Processing country {filename}, from chapter {chapters_in_country[filename]['start']} to chapter {chapters_in_country[filename]['end']}")
    whole_prompt = instruction + "\n"
    start = chapters_in_country[filename]['start']
    end = chapters_in_country[filename]['end']
    for i in range(start, end+1):
        chapter = helpers.get_chapter(i, False)
        prompt += chapter
        whole_prompt += chapter

    #if not os.path.isdir("dump"):
    #    os.mkdir("dump")
    #with open(f"dump/whole_{filename}.txt", "w", encoding="utf-8") as file:
    #    file.write(whole_prompt)
    #with open(f"dump/{filename}.txt", "w", encoding="utf-8") as file:
    #    file.write(prompt)
    
    # Check for number of tokens sent in the last minute
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(whole_prompt))
    print(num_tokens)
    
    time_to_wait = last_request - time.time() + 60
    # Don't wait before sending the first query
    if not first:
        if time_to_wait < 0:
            print(f" {time_to_wait} seconds passed since last request, no need to wait, sending request now")
        else:
            if time_to_wait < 59.5:
                print(f"Send only one request per minute, wait for {time_to_wait} seconds")
                time.sleep(time_to_wait)
    first = False
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
    if not os.path.isfile(f"output/GPT/tokens/tokens_country_summary.csv"):
        with open("output/GPT/tokens/tokens_country_summary.csv", "w", encoding='utf-8') as file:
            file.write("Country;Prompt;Completion;Total\n")

    with open("output/GPT/tokens/tokens_country_summary.csv", "a", encoding="utf-8") as file:
        file.write(f"{filename};{prompt_tokens};{completion_tokens};{total_tokens}\n")
    #print(message)

    # Put all main locations to single file
    #locations = parts[0]
    with open(f"output/GPT/summary/{filename}.txt", "w", encoding="utf-8") as file:
        file.write(message)
'''