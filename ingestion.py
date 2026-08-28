from openai import OpenAI
from pathlib import Path
import json



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

def load_or_create_embeddings(
    chunks: list[str], 
    client: OpenAI,
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
