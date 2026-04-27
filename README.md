**Task 1. Image classification + OOP**

We`ve built 3 models:
* Random Forest;
* Feed-Forward Neural Network;
* Convolutional Neural Network;

Each model is a separate class that implements ***MnistClassifierInterface*** with 2 abstract methods - ***train*** and ***predict***. Every model is hidden under another MnistClassifier class. MnistClassifer takes an ***algorithm*** as an input parameter. The values for the algorithm are: ***cnn***, ***rf***, and ***ffn*** respectively.

**Task 2. Named entity recognition + image classification**

We have developed a system for cross-modal verification between images and text. 
The task utilizes two specialized models:
* CNN (ResNet50): responsible for classifying animal species in images.
* NER (DistilBERT): used to extract animal mentions from text sentences.

Both models are integrated into a single ***VerificationPipeline*** class. The system not only matches the predicted image class with the entities extracted from the text but also analyzes the linguistic context for negations (e.g., "not a dog") to prevent false positives. The architecture is built using interfaces (***ImageClassifierInterface*** and ***NerInterface***), ensuring modularity and allowing for easy component replacement.
