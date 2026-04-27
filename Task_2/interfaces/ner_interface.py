from abc import ABC, abstractmethod

class NerInterface(ABC):

    @abstractmethod
    def train(self, dataset_path: str, **kwargs):
        pass

    @abstractmethod
    def predict(self, text: str, **kwargs) -> list[str]:
        pass