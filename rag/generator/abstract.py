from abc import abstractmethod
from typing import Any, Dict, Generator, List

from .prompt import Prompt


class AbstractGenerator(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @abstractmethod
    def generate(self, prompt: Prompt) -> Generator[Any, Any, Any]:
        pass

    @abstractmethod
    def chat(
        self, prompt: Prompt, messages: List[Dict[str, str]]
    ) -> Generator[Any, Any, Any]:
        pass
