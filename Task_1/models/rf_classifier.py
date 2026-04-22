from sklearn.ensemble import RandomForestClassifier as rf_classifier
from interfaces.mnist_interface import MnistClassifierInterface

class RandomForestClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.model = rf_classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

    def train(self, x_train, y_train):
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train_flat, y_train)

    def predict(self, x_test):
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        return self.model.predict(x_test_flat)