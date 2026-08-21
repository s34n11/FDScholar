from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path # built-in tool for working with file and folder paths
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

book_path = Path("data/notes_from_underground.txt") 

text = book_path.read_text(encoding="utf-8")

print(text[:1000])
print("\nCharacter count:", len(text))
