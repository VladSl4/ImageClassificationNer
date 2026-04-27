from abc import ABC, abstractmethod

class ImageClassifierInterface(ABC):

    @abstractmethod
    def train(self, dataset_path: str, **kwargs):
        pass

    @abstractmethod
    def predict(self, image_path: str) -> str:
        pass
