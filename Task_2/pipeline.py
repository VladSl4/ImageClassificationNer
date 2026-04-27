import re
from typing import List
from interfaces.cv_interface import ImageClassifierInterface
from interfaces.ner_interface import NerInterface

class VerificationPipeline:
    def __init__(self, cv_model: ImageClassifierInterface, nlp_model: NerInterface):
        self.cv_model = cv_model
        self.nlp_model = nlp_model
        self.negation_words = {"no", "not", "isn't", "aren't", "doesn't", "don't", "never", "without", "except"}

    def verify(self, image_path: str, text: str) -> bool:
        image_animal = self.cv_model.predict(image_path)
        text_animals = self.nlp_model.predict(text)

        if not image_animal or not text_animals:
            return False

        image_animal = image_animal.lower()
        text_animals = [a.lower() for a in text_animals]

        if image_animal not in text_animals:
            return False

        words = re.findall(r'\b\w+\b', text.lower())

        animal_indices = [i for i, word in enumerate(words) if word == image_animal]

        for index in animal_indices:
            start_window = max(0, index - 3)
            preceding_words = set(words[start_window:index])
            if self.negation_words.intersection(preceding_words):
                continue
            else:
                return True

        return False