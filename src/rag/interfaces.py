from abc import ABC, abstractmethod

class IRAG(ABC):
    
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def query(self, prompt: str):
        pass