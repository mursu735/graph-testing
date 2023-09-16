import os

import openai

key = ""
with open("key.txt") as file:
    key = file.read()
print(key)

openai.api_key = key

# list models
#models = openai.Model.list()

# print the first model's id
#print(models.data)

instruction = ""
with open("input/instruction.txt") as file:
    instruction = ''.join(line.rstrip() for line in file)

prompt = ""
with open("input/prompt.txt") as file:
    prompt = ''.join(line.rstrip() for line in file)

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}])

# print the chat completion
print(chat_completion.choices[0].message.content)


with open("output/output.txt", "w") as file:
   file.write(chat_completion.choices[0].message.content)
