from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseProcessor(ABC):
    def process(self):
        pass
