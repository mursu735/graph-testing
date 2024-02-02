import os
import re
import time
import openai
import tiktoken
import helpers


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def determine(file):
    filename = file.split(".")
    filename = filename[0]
    return os.path.isdir(f"output/GPT/character_summaries/{filename}")

key = ""
with open("key.txt") as file:
    key = file.read()
print(key)

openai.api_key = key

# list models
#models = openai.Model.list()

# print the first model's id
#print(models.data)

instruction = helpers.get_summary_instruction()
print(instruction)

if not os.path.exists("output/GPT"):
    os.mkdir("output/GPT")
    
#if not os.path.isfile("output/GPT/locations.csv"):
#   with open("output/GPT/locations.csv", "w", encoding="utf-8") as file:
#       file.write("Location;Latitude;Longitude;Order\n")

if not os.path.exists("output/GPT/character_summaries"):
    os.mkdir("output/GPT/character_summaries")


prompt = ""
files = os.listdir("input/Chapters")
files = natural_sort(files)
#print("Before:")
#print(files)
# Only include chapters that haven't been parsed yet to reduce load and cost
files = [file for file in files if not determine(file)]
first = True
print(f"Processing files {files}")

last_request = time.time()

for file in files:
    filename = int(file.split(".")[0])
    print(f"Processing chapter {filename}")
    #with open(f"input/Chapters/{file}", encoding='utf-8') as file:
    prompt = helpers.get_chapter(filename)

    # Check for number of tokens sent in the last minute
    whole_prompt = instruction + "\n" + prompt
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(whole_prompt))
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
    if not os.path.isfile(f"output/GPT/tokens/tokens_summary.csv"):
        with open("output/GPT/tokens/tokens_summary.csv", "w", encoding='utf-8') as file:
            file.write("Chapter;Prompt;Completion;Total\n")

    with open("output/GPT/tokens/tokens_summary.csv", "a", encoding="utf-8") as file:
        file.write(f"{filename};{prompt_tokens};{completion_tokens};{total_tokens}\n")
    #print(message)

    parts = message.split("///")
    print(parts)
    # Put all main locations to single file
    #locations = parts[0]
    #with open("output/GPT/locations.csv", "a", encoding="utf-8") as file:
    #    file.write(locations)
    for summary in parts:
        split = summary.split(":")
        character = split[0].strip()
        text = split[1].strip()
        print("Summary", summary)
        print("Character", character)
        print("Text", text)

        if not os.path.isdir(f"output/GPT/character_summaries/{filename}"):
            os.mkdir(f"output/GPT/character_summaries/{filename}")
        # Put each character's summary to a different file
        with open(f"output/GPT/character_summaries/{filename}/{character}.txt", "w", encoding="utf-8") as file:
            file.write(text)
        #summary = parts[3]

    
#print(prompt)

'''
print(files)

for file in files:
    with open(f"input/texts/{file}", encoding='utf-8') as f:
        prompt = prompt + ''.join(line for line in f)

print(prompt)
with open("testing.txt", "w", encoding='utf-8') as file:
    file.write(prompt)
'''


#with open("output/output_test.csv", "w") as file:
#   file.write(chat_completion.choices[0].message.content)
