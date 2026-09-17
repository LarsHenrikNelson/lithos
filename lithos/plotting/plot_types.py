from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BaseProcessor(ABC):
    def process(self):
        pass
