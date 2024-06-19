from typing import Any, Generator, List

from loguru import logger as log

from rag.generator import get_generator
from rag.generator.prompt import Prompt
from rag.message import Message
from rag.retriever.rerank import get_reranker
from rag.retriever.retriever import Retriever
from rag.retriever.vector import Document


class Rag:
    def __init__(self, client: str = "local") -> None:
        self.bot = None
        self.user = None
        self.client = client
        self.messages: List[Message] = []
        self.retriever = Retriever()
        self.reranker = get_reranker(self.client)
        self.generator = get_generator(self.client)
        self.__set_roles()

    def __set_roles(self):
        self.bot = "assistant" if self.client == "local" else "CHATBOT"
        self.user = "user" if self.client == "local" else "USER"

    def set_client(self, client: str):
        log.info(f"Setting client to {client}")
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
        self.messages.append(Message(role=role, content=content, client=self.client))

    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        if self.messages:
            messages = self.reranker.rerank_messages(prompt.query, self.messages)
        else:
            messages = []
        messages.append(
            Message(role=self.user, content=prompt.to_str(), client=self.client)
        )
        return self.generator.generate(messages, prompt.documents)
