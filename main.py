from main import embeddings_path
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path # built-in tool for working with file and folder paths
import math
import json
import os

load_dotenv() # looks for an .env file and loads the key-value pairs in 
                # it into your program's environment variables



# LOAD UP THE BOOK

def load_book(book_path: str) -> str:
    """
    Return the book text based on the book path.
    """
    text = Path(book_path).read_text(encoding="utf-8")

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    # finds the index of the start marker
    start = text.find(start_marker)
    end = text.find(end_marker)

    if start != -1:
        start = text.find("\n", start) # find the newline from the start marker
        text = text[start + 1:]

    # find the index of the end marker
    end = text.find(end_marker)

    if end != -1:
        text = text[:end]

    return text.strip()



# RAG IMPLEMENTATION

client = OpenAI()

# break book down into manageable checks which will later be retrieved depending on the prompt

def chunk_text(text: str, chunk_size: int=1000, overlap: int=200) -> list[str]:
    """
    Convert the book text into chunks. 
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # ensures that subsequent chunks do not lose their context captured in preceding chunks
        start = end - overlap

    return chunks


# check if the embeddings for each chunk exists:

def load_or_create_embeddings(chunks: list[str], 
    embeddings_path: str="chunk_embeddings.json") -> list[list]:
    """
    Load or create embeddings for the chunks of the given book.
    """

    embeddings_path = Path(embeddings_path)

    if embeddings_path.exists():
        with open(embeddings_path, "r") as f:
            chunk_embeddings = json.load(f)

    else:

    # create an embedding for each chunk (if it does not exist)

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks
        )

        chunk_embeddings = [
            item.embedding 
            for item in response.data
        ]


        with open(embeddings_path, "w") as f:
            json.dump(chunk_embeddings, f)


    return chunk_embeddings


# create embedding for the question

def embed_question(question: str) -> list[float]:
    """
    Create embedding for the user's question.
    """
    response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
    )

    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Measures the similarity between two embeddings.
    """
    dot_product = sum(x * y for x, y in zip(a,b)) # iterates through all 'dimension pairs' of the vectors a,b
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (mag_a * mag_b)


def retrieve_chunks(question_embedding, chunk_embeddings, chunks, top_k=3) -> list[tuple]:
    """
    Finds the top_k most relevant chunks for the given question embedding.
    """

    similarities = []

    for i, embedding in enumerate(chunk_embeddings): # the fxn enumerate gives you both the index and the embedding
        score = cosine_similarity(question_embedding, embedding)
        similarities.append((score, i))

    similarities.sort(reverse=True)

    # retrieve the most similar chunks relative to the question
    top_chunks = []

    for score, i in similarities[:top_k]:
        top_chunks.append((score, i, chunks[i]))

    return top_chunks


def build_context(top_chunks: list[tuple]) -> str:
    """
    Builds the context for the language model's answer.
    """
    context_parts = []

    for score, i, chunk in top_chunks:
        context_parts.append(
            f"[Source {i}]\n{chunk}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question: str, context: str) -> str:
    """
    Generate's the language model's answer to the user's question.
    """

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=f"""
        You are answering questions about Notes from Underground by Dostoyevsky.

        Use the retrieved passages below as your primary evidence.

        When making claims supported by a retrieved passage, cite the relevant source using its label, e.g. '[Source 147]'

        If the passages do not contain enough information to answer confidently, say so.

        Question:
        {question}

        Retrieved Passages:
        {context}
        """
    )

    return response.output_text


# QUESTION_ANSWER FLOW:

text = load_book("data/notes_from_underground.txt")
chunks = chunk_text(text)
chunk_embeddings = load_or_create_embeddings(chunks)

question = input("Any questions about the book? ")
question_embedding = embed_question(question)

top_chunks = retrieve_chunks(question_embedding, chunk_embeddings, chunks)
context = build_context(top_chunks)
answer = generate_answer(question, context)

print(answer)