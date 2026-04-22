**Task 1. Image classification + OOP**

We`ve built 3 models:
* Random Forest;
* Feed-Forward Neural Network;
* Convolutional Neural Network;

Each model is a separate class that implements ***MnistClassifierInterface*** with 2 abstract methods - ***train*** and ***predict***. Every model is hidden under another MnistClassifier class. MnistClassifer takes an ***algorithm*** as an input parameter. The values for the algorithm are: ***cnn***, ***rf***, and ***nn*** respectively.
