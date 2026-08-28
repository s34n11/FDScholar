from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path # built-in tool for working with file and folder paths
import math
import json

from ingestion import load_book, chunk_text, load_or_create_embeddings
from retrieval import embed_question, retrieve_chunks
from generation import build_context, generate_answer

load_dotenv() # looks for an .env file and loads the key-value pairs in 
                # it into your program's environment variables

client = OpenAI()

# QUESTION_ANSWER FLOW:

text = load_book("data/notes_from_underground.txt")
chunks = chunk_text(text)
chunk_embeddings = load_or_create_embeddings(chunks, client=client)

question = input("Any questions about the book? ")
question_embedding = embed_question(question, client=client)

top_chunks = retrieve_chunks(question_embedding, chunk_embeddings, chunks)
context = build_context(top_chunks)
answer = generate_answer(question, context, client)

print(answer)