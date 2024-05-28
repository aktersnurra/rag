from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Log:
    user: Message
    bot: Message

    def get():
        return (user, bot)


@dataclass
class Message:
    role: str
    message: str

    def as_dict(self, model: str) -> Dict[str, str]:
        if model == "cohere":
            match self.role:
                case "user":
                    role = "USER"
                case _:
                    role = "CHATBOT"

            return {"role": role, "message": self.message}
        else:
            return {"role": self.role, "content": self.message}


class Memory:
    def __init__(self, reranker) -> None:
        self.history = []
        self.reranker = reranker
        self.user = "user"
        self.bot = "assistant"

    def add(self, prompt: str, response: str):
        self.history.append(
            Log(
                user=Message(role=self.user, message=prompt),
                bot=Message(role=self.bot, message=response),
            )
        )

    def get(self) -> List[Log]:
        return [m.as_dict() for log in self.history for m in log.get()]

    def reset(self):
        self.history = []
