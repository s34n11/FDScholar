import math
from openai import OpenAI


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Measures the similarity between two embeddings.
    """
    dot_product = sum(x * y for x, y in zip(a,b)) # iterates through all 'dimension pairs' of the vectors a,b
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (mag_a * mag_b)

# create embedding for the question

def embed_question(question: str,
client: OpenAI) -> list[float]:
    """
    Create embedding for the user's question.
    """
    response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
    )

    return response.data[0].embedding


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