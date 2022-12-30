from statistics import mode
from typing import Any, Callable


class ContextReader:
    def __init__(
        self,
        name: str,
        window_length: int,
        read_from_context: Callable[[dict], Any],
        value_mutator: Callable[[Any], Any] = lambda x: x,
    ):
        self.name = name
        self.window_length = window_length
        self.history = []
        self.read_from_context = read_from_context
        self.value_mutator = value_mutator

    def read(self, context: dict):
        self.history.append(self.read_from_context(context))
        self.history = self.history[-self.window_length :]

    def get_value(self):
        return self.value_mutator(mode(self.history))
