from dataclasses import dataclass
from typing import Dict


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
