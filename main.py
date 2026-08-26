from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path # built-in tool for working with file and folder paths
import math
import json
import os

load_dotenv() # looks for an .env file and loads the key-value pairs in 
                # it into your program's environment variables

# test_key = os.getenv("TEST_KEY")
# print(test_key)


# client = OpenAI()

# response = client.responses.create(
#     model="gpt-5.6-luna",
#     input="Say hello in one sentence"
# )

# print(response.output_text)




# LOAD UP THE BOOK

book_path = Path("data/notes_from_underground.txt") 

text = book_path.read_text(encoding="utf-8")

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK NOTES FROM THE UNDERGROUND ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK NOTES FROM THE UNDERGROUND ***"

# finds the index of the start and end markers
start = text.find(start_marker)
end = text.find(end_marker)

if start != -1:
    text = text[start + len(start_marker):]

if end != -1:
    text = text[:end]








# RAG IMPLEMENTATION

# break book down into manageable checks which will later be retrieved depending on the prompt

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # ensures that subsequent chunks do not lose their context captured in preceding chunks
        start = end - overlap

    return chunks

chunks = chunk_text(text)

# check if the embeddings for each chunk exists:

embeddings_path = Path("chunk_embeddings.json")

if embeddings_path.exists():

    with open(embeddings_path, "r") as f:
        chunk_embeddings = json.load(f)

else:

# create an embedding for each chunk (if it does not exist)

    client = OpenAI()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    chunk_embeddings = [
        item.embedding 
        for item in response.data
    ]

    print("Number of embeddings:", len(chunk_embeddings))

    with open(embeddings_path, "w") as f:
        json.dump(chunk_embeddings, f)


# sample question 1:

q1 = "why does the underground man act against his own interests?"

q1_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=q1
)

q1_embedding = q1_response.data[0].embedding

# compare q1 embedding to each chunk's embedding


# measures the similarity of two embeddings
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a,b)) # iterates through all 'dimension pairs' of the vectors a,b
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (mag_a * mag_b)


similarities = []

for i, embedding in enumerate(chunk_embeddings): # the fxn enumerate gives you both the index and the embedding
    score = cosine_similarity(q1_embedding, embedding)
    similarities.append((score, i))

similarities.sort(reverse=True)



# generate a response: take the most similar chunks and send them to the language model 
# along with the user's question

# retrieve the most similar chunks relative to the question
top_chunks = []

for score, i in similarities[:3]:
   top_chunks.append(chunks[i])

context = "\n\n---\n\n".join(top_chunks)

response = client.responses.create(
    model="gpt-5.6-luna",
    input=f"""
    You are answering questions about Notes from Underground by Dostoyevsky.

    Use the retrieved passages below as your primary evidence.
    If the passages do not contain enough information to answer confidently, say so.

    Question:
    {q1}

    Retrieved Passages:
    {context}
    """
)

print(response.output_text)
