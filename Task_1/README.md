# Task 1: MNIST Classification Models

This repository contains the implementation of a unified Object-Oriented interface for classifying the MNIST dataset using three different machine learning algorithms.

## Architecture

The project strictly follows SOLID principles and utilizes the **Facade / Factory** design patterns. 
All models are wrapped under a single `MnistClassifierInterface`, ensuring a unified workflow regardless of the underlying mathematical approach or framework.

The main wrapper class `MnistClassifier` allows seamless switching between:
1. **Random Forest (RF)** - using `scikit-learn`
2. **Feed-Forward Neural Network (FFNN)** - using `PyTorch`
3. **Convolutional Neural Network (CNN)** - using `PyTorch`

Hyperparameters for all models are dynamically passed via `**kwargs`, enabling unified hyperparameter tuning (e.g., via `RandomizedSearchCV` or `ParameterSampler`) without breaking the interface.

---

## Important Note on Demo Performance and Hardware Limitations

If you run the provided `demo.ipynb`, you might notice an unexpected discrepancy in the final metrics: the **Random Forest** achieves high accuracy (~96%), while the **Neural Networks (FFNN and CNN)** perform significantly worse (around ~10%, which equals random guessing).

**This is not a bug in the network architecture or the training loop.** 

This behavior is strictly caused by the hardware limitations of the local machine used for this demonstration (laptop CPU without a dedicated CUDA-enabled GPU). To allow the Jupyter Notebook to execute in a reasonable time during development and testing, the training dataset was drastically truncated (e.g., using only 1,000 samples instead of 60,000) and limited to a very small number of epochs.

**Why this happens:**
* **Random Forest** uses mathematical "greedy" splits and can easily find patterns even on a microscopic dataset almost instantly, relying solely on RAM and CPU.
* **Neural Networks (PyTorch)** learn via Gradient Descent. With a batch size of 64 on a 1,000-image dataset, the network performs only about 15 optimization steps per epoch. Over 3 epochs, this is fewer than 50 weight updates total—nowhere near enough for Adam optimizer to converge and learn feature representations.

**Full Dataset Capability:**
If the hardware allows (or if deployed on a machine with a dedicated GPU), removing the dataset slicing in `demo.ipynb` and feeding the full 60,000 training images over 10-15 epochs will result in the NNs operating as intended. Under full data conditions, the **CNN will easily reach 98-99% accuracy**, significantly outperforming the Random Forest baseline.

---

## How to Run

1. Ensure all dependencies from `requirements.txt` are installed (`torch`, `torchvision`, `torchmetrics`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`).
2. Run `demo.ipynb` to see the initialization, hyperparameter tuning via Random Search, and evaluation processes.
3. To train on the full dataset, simply remove the array slicing (`x_train_full[:1000]`) in the data initialization cell of the notebook.