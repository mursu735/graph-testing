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

instruction = """Given a prompt, extrapolate as many relationships as possible from it and provide a list of updates.

If an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.

If an update is related to a color, provide [ENTITY, COLOR]. Color is in hex format.

If an update is related to deleting an entity, provide ["DELETE", ENTITY].

Example:
prompt: Alice is Bob's roommate. Make her node green.
updates:
[["Alice", "roommate", "Bob"], ["Alice", "#00FF00"]]"""

prompt = """
prompt: Ishmael: The novel's narrator and protagonist, Ishmael is a young sailor who seeks adventure and a deeper understanding of life. He joins the crew of the Pequod and provides insightful observations about the crew, whaling, and philosophical reflections.  Captain Ahab: The driven and obsessed captain of the Pequod, Ahab's sole focus is to hunt down and kill the white whale, Moby Dick, that took his leg. His monomaniacal pursuit of vengeance becomes central to the story's themes of revenge and the destructive power of obsession.  Moby Dick: The infamous white whale that is the object of Ahab's obsession. Moby Dick is a symbol of the uncontrollable forces of nature and the unknown.  Starbuck: The first mate of the Pequod, Starbuck is a thoughtful and moral man who often questions Ahab's obsessive quest. He represents reason and caution in contrast to Ahab's reckless pursuit of vengeance.  Queequeg: A harpooner and Ishmael's close friend, Queequeg is a skilled and fearsome warrior from the South Seas. He serves as a symbol of friendship and camaraderie, transcending cultural differences.  Stubb: The second mate of the Pequod, Stubb is known for his humor and laid-back demeanor. He provides comic relief in the midst of the tense atmosphere on the ship.  Tashtego and Daggoo: Harpooners of the Pequod, Tashtego is a Native American and Daggoo is a tall African man. Both characters highlight the diversity of the crew and the camaraderie that develops among them.  Fedallah: A mysterious, enigmatic figure, Fedallah is Ahab's personal harpooner. He has a prophetic aura and becomes associated with Ahab's fate.  Peleg and Bildad: The owners of the Pequod, Peleg and Bildad represent the business side of whaling. They hire Ishmael and Queequeg and provide insight into the commercial aspects of the industry."""

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}])

# print the chat completion
print(chat_completion.choices[0].message.content)