import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("vocab size:", encoder.n_vocab)

text = "my name is talha jubair"

tokens = encoder.encode(text)
print("tokens:", tokens)

decoder = encoder.decode(tokens)
print("decoded:", decoder)