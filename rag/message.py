from dataclasses import dataclass
from typing import Dict, List

from loguru import logger as log


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


@dataclass
class Messages:
    messages: List[Message]

    def __add__(self, message: Message):
        self.messages.append(message)

    def __len__(self):
        return len(self.messages)

    def reset(self):
        log.debug("Resetting messages...")
        self.messages = []

    def content(self) -> List[str]:
        return [m.content for m in self.messages]

    def rerank(self, rankings: List[int]):
        self.messages = [self.messages[r] for r in rankings]
