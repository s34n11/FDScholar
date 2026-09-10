from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path # built-in tool for working with file and folder paths
import math
import json

from book_config import NOTES_FROM_UNDERGROUND
from ingestion import load_book, chunk_text, split_into_chapters, load_or_create_embeddings
from retrieval import embed_question, retrieve_chunks
from generation import build_context, generate_answer



load_dotenv() # looks for an .env file and loads the key-value pairs in 
                # it into your program's environment variables

client = OpenAI()

# QUESTION_ANSWER FLOW:

curr_book = NOTES_FROM_UNDERGROUND

text = load_book(curr_book["path"])

chapters = split_into_chapters(
    text,
    curr_book["part_headings"],
    curr_book["chapter_headings"]
)

chunks = []

for chapter in chapters:
    chapter_chunks = chunk_text(chapter["text"])

    for chunk in chapter_chunks:
        chunks.append({
            "text": chunk,
            "part": chapter["part"],
            "chapter": chapter["current_chapter"]
        })

chunk_embeddings = load_or_create_embeddings(chunks, client=client)

question = input("Any questions about the book? ")
question_embedding = embed_question(question, client=client)

top_chunks = retrieve_chunks(question_embedding, chunk_embeddings, chunks)
context = build_context(top_chunks)

answer = generate_answer(question, context, client)

print(answer)