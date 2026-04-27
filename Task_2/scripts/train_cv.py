import argparse
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from models.animal_cv_classifier import AnimalClassifier


def main():
    parser = argparse.ArgumentParser(description="Train CV Model (ResNet50)")

    data_group = parser.add_argument_group('Path settings')
    data_group.add_argument("--data", type=str, required=True,
                            help="Dataset folder path (e.g., data/animals-10)")
    data_group.add_argument("--save_path", type=str, default="weights/best_cv_model.pth",
                            help="Weights save path (e.g., weights/best_cv_model.pth)")

    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    hyper_group.add_argument("--batch_size", type=int, default=32, help="Batch size")
    hyper_group.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    print("=== Start Computer Vision training pipeline ===")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.num_epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    model = AnimalClassifier()

    model.train(
        dataset_path=args.data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )

    print("=== Training completed successfully ===")


if __name__ == "__main__":
    main()