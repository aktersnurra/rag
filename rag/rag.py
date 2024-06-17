from dataclasses import dataclass
from typing import Any, Dict, Generator, List

from loguru import logger as log

from rag.generator import get_generator
from rag.generator.prompt import Prompt
from rag.retriever.rerank import get_reranker
from rag.retriever.retriever import Retriever
from rag.retriever.vector import Document


@dataclass
class Message:
    role: str
    content: str
    client: str

    def as_dict(self) -> Dict[str, str]:
        if self.client == "cohere":
            return {"role": self.role, "message": self.content}
        else:
            return {"role": self.role, "content": self.content}


class Rag:
    def __init__(self, client: str) -> None:
        self.messages: List[Message] = []
        self.retriever = Retriever()
        self.client = client
        self.reranker = get_reranker(self.client)
        self.generator = get_generator(self.client)
        self.bot = "assistant" if self.client == "ollama" else "CHATBOT"
        self.user = "user" if self.client == "ollama" else "USER"

    def __set_roles(self):
        self.bot = "assistant" if self.client == "ollama" else "CHATBOT"
        self.user = "user" if self.client == "ollama" else "USER"

    def set_client(self, client: str):
        self.client = client
        self.reranker = get_reranker(self.client)
        self.generator = get_generator(self.client)
        self.__set_roles()
        self.__reset_messages()
        log.debug(f"Swapped client to {self.client}")

    def __reset_messages(self):
        log.debug("Deleting messages...")
        self.messages = []

    def retrieve(self, query: str) -> List[Document]:
        documents = self.retriever.retrieve(query)
        log.info(f"Found {len(documents)} relevant documents")
        return self.reranker.rerank_documents(query, documents)

    def add_message(self, role: str, content: str):
        self.messages.append(
            Message(role=role, content=content, client=self.client)
        )

    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        messages = self.reranker.rerank_messages(prompt.query, self.messages)
        self.messages.append(
            Message(
                role=self.user, content=prompt.to_str(), client=self.client
            )
        )
        return self.generator.generate(prompt, messages)
