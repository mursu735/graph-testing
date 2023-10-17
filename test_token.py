import tiktoken


# To get the tokeniser corresponding to a specific model in the OpenAI API:


with open("token.txt", encoding="utf-8") as file:
    text = file.read()
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(text))
    print(num_tokens)