from models.rf_classifier import RandomForestClassifier as rf
from models.ffn_classifier import FeedForwardClassifier as ffn
from models.cnn_classifier import ConvolutionalClassifier as cnn
from interfaces.mnist_interface import MnistClassifierInterface

class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm: str, **kwargs):
        self.algorithm = algorithm.lower()
        self.kwargs = kwargs

        if self.algorithm == 'rf':
            self.classifier = rf(**self.kwargs)
        elif self.algorithm == 'ffn':
            self.classifier = ffn(**self.kwargs)
        elif self.algorithm == 'cnn':
            self.classifier = cnn(**self.kwargs)
        else:
            raise ValueError(
                f'Algorithm {self.algorithm} is not implemented. Choose from: "rf", "ffn", "cnn".'
            )

    def train(self, x_train, y_train):
        self.classifier.train(x_train, y_train)

    def predict(self, x_test):
        return self.classifier.predict(x_test)