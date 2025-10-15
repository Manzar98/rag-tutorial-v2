import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "mistral"

class OllamaEmbeddingFunction:
    """Custom embedding function compatible with LangChain and Chroma."""

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            payload = {"model": MODEL, "prompt": text}
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])
        return embeddings

    def embed_query(self, text):
        payload = {"model": MODEL, "prompt": text}
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["embedding"]


def get_embedding_function():
    return OllamaEmbeddingFunction()
