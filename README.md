
# FDScholar

FDScholar is a Retrieval-Augmented Generation (RAG) assistant designed to help readers explore and better understand the literary and philosophical works of Fyodor Dostoyevsky.

Currently, the project focuses on *Notes from Underground*, allowing users to ask questions about the text and receive contextually relevant, AI-generated responses grounded in the novel.

## How It Works

FDScholar uses a RAG pipeline to retrieve relevant passages from the text before generating an answer.

1. **Text Processing:** Loads and splits the novel into overlapping text chunks, preserving chapter information.
2. **Embedding Generation:** Converts text chunks into vector embeddings using OpenAI's `text-embedding-3-small` model.
3. **Semantic Retrieval:** Embeds user questions and uses cosine similarity to identify the most relevant passages.
4. **Response Generation:** Passes the retrieved passages and the user's question to an OpenAI language model to generate a context-aware response.

Embeddings are cached locally to reduce redundant API calls.

## Tech Stack

- Python
- OpenAI API
- Vector embeddings and cosine similarity
- Retrieval-Augmented Generation (RAG)

## Current Status

**Work in Progress**

The core RAG pipeline has been implemented for *Notes from Underground*. Ongoing development focuses on evaluating retrieval quality and improving the assistant's responses.

### Planned Features

- Retrieval and response evaluation
- Support for additional Dostoyevsky novels
- A simple interactive user interface
- Conversation history for follow-up questions

## Project Goal

To build an AI-powered literary research assistant that helps readers engage more deeply with Dostoyevsky's works through text-grounded explanations, summaries, and philosophical analysis.
