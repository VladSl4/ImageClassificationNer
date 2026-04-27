# Task 2: Animal Recognition Verification Pipeline

This repository contains the implementation of a robust, end-to-end Machine Learning pipeline designed to verify animal entities by combining Computer Vision (CV) and Natural Language Processing (NLP).

## Architecture

The project strictly adheres to SOLID principles, heavily utilizing **Dependency Inversion** and the **Strategy** design pattern. The core logic is orchestrated by a unified `VerificationPipeline`, which acts as a controller that accepts distinct, isolated models through interfaces (`ImageClassifierInterface` and `NerInterface`).

This decoupled architecture ensures that the CV or NLP components can be upgraded, swapped, or retrained entirely independently without affecting the overall verification logic.

The pipeline comprises three main components:
1. **Computer Vision Model (ResNet50)** - using `PyTorch` and `torchvision` to classify images into 10 animal categories.
2. **Named Entity Recognition Model (DistilBERT)** - using Hugging Face `transformers` and `datasets` to extract animal entities from text using BIO tagging.
3. **Verification Orchestrator** - A custom local-window proximity checking algorithm that combines CV and NLP predictions, handling edge cases such as linguistic negations (e.g., "not a dog") and sentences containing multiple entities.

Training and inference scripts are completely parameterized using `argparse`, making them production-ready for MLOps workflows.

---

## Important Note on Dataset, Weights, and Environments

This task involves two distinct deep learning domains (Vision and Text), both requiring significant computational resources for training. 

**1. Computer Vision (ResNet50):**
The CV model utilizes Transfer Learning on a ResNet50 backbone. While inference is extremely fast and can run locally on a CPU, training on the full `Animals-10` dataset requires a GPU for efficient execution. The provided parameterized script (`scripts/train_cv.py`) is designed to be executed in cloud environments (e.g., Google Colab with T4 GPUs). 

**2. NLP / NER (DistilBERT):**
Training the Token Classification model for NER requires fine-tuning a pre-trained Transformer. This process is highly memory-intensive and is strictly intended to be run on GPU instances (`scripts/train_ner.py`). The Hugging Face `Trainer` API handles the optimization, but local CPU training is highly discouraged due to time constraints.

**The Orchestrator (`VerificationPipeline`):**
The final verification logic and the `demo.ipynb` notebook do **not** require training from scratch. They are designed to load pre-trained weights/checkpoints and perform *inference only*. If you wish to test the pipeline locally without training the models, ensure that the trained weights are correctly placed in the `weights/` directory as specified in the scripts.

---

## How to Run

1. Ensure all dependencies from `requirements.txt` are installed (`torch`, `torchvision`, `transformers`, `datasets`, `pandas`, `numpy`, `matplotlib`, `seaborn`).

2. **For Training (Recommended on GPU):**
   * **CV Model:** `!python scripts/train_cv.py \
    --data /content/data/processed_images/train \
    --save_path /content/weights/best_cv_model.pth \
    --num_epochs 5`
   * **NER Model:** `!python scripts/train_ner.py \
    --data /content/data/ner_dataset.json \
    --save_dir /content/weights/best_ner_model`

3. **For Independent Inference:**
   * **CV Model:** `!python3 scripts/inference_cv.py --image "$TEST_IMAGE_PATH" --weights "weights/best_cv_model.pth"`
   * **NER Model:** `!python3 scripts/inference_ner.py --text "$TEST_TEXT" --model_dir "weights/best_ner_model"`

4. **For the Full Pipeline & EDA:**
   Run `EdaAndDemo.ipynb` to see the Exploratory Data Analysis (EDA) for both datasets and a demonstration of the `VerificationPipeline` resolving complex linguistic edge cases.

**NOTE**
Due to hardware limitations of the local machine, the model training process for ***Task 2*** was conducted using Google Colab to leverage its T4 GPU acceleration. This allowed for more efficient training of the ResNet50 and DistilBERT architectures.
