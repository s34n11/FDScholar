from openai import OpenAI

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


def generate_answer(question: str, 
context: str,
client: OpenAI) -> str:
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
