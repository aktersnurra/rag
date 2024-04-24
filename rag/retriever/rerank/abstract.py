from abc import abstractmethod

from rag.generator.prompt import Prompt


class AbstractReranker(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @abstractmethod
    def rank(self, prompt: Prompt) -> Prompt:
        return prompt
