import openai
from typing import List
import os
from dotenv import load_dotenv
import yaml

load_dotenv()

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str = config['embedding']['model']):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.batch_size = config['embedding']['batch_size']

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)

def get_embedding_function(api_key: str):
    return OpenAIEmbeddings(api_key)