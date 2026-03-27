from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings


class OllamaProvider:
    def __init__(self, base_url: str, model: str, embedding_model: str) -> None:
        self.chat_model = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.1,
            num_predict=256,
            client_kwargs={"timeout": 180},
        )
        self.embedding_model = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url,
            client_kwargs={"timeout": 180},
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.embed_documents(texts)

    def get_embedding_function(self) -> OllamaEmbeddings:
        return self.embedding_model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self.chat_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return str(response.content)
