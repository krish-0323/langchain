from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(models='text-embedding-3-large', dimensions=32)

# Generating embedding for single query

# result = embedding.embed_query("Delhi is the capital of India")
# print(str(result))

# Generating embedding for mutiple query as documents

documents = [
    "Delhi is the capital city of India",
    "Lucknow is the capital of Uttar Pradesh",
    "Paris is the capital city of France"
]

result = embedding.embed_documents(documents)

print(str(result))