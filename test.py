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
with open("input/instruction_location.txt") as file:
    instruction = ''.join(line.rstrip() for line in file)

prompt = ""

files = os.listdir("input/texts")

with open("input/prompt.txt", encoding='utf-8') as file:
    prompt = file.read()

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
# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}])
#chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}])

# print the chat completion
print(chat_completion.choices[0].message.content)


with open("output/output_test.csv", "w") as file:
   file.write(chat_completion.choices[0].message.content)
