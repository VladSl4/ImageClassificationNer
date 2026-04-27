import argparse
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from models.animal_ner_model import AnimalNerModel


def main():
    parser = argparse.ArgumentParser(description="Train NER model (DistilBERT)")

    data_group = parser.add_argument_group('Path settings')
    data_group.add_argument("--data", type=str, required=True,
                            help="JSON dataset path (e.g., data/ner_dataset.json)")
    data_group.add_argument("--save_dir", type=str, default="weights/best_ner_model",
                            help="Folder for saving weights (e.g., weights/best_ner_model)")

    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument("--num_epochs", type=int, default=3,
                             help="Number of epochs")
    hyper_group.add_argument("--batch_size", type=int, default=16, help="Batch size")
    hyper_group.add_argument("--lr", type=float, default=2e-5,
                             help="Learning rate")

    args = parser.parse_args()

    print("=== Starting NER training pipeline ===")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.num_epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    model = AnimalNerModel()

    model.train(
        dataset_path=args.data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_dir
    )

    print("=== Training completed successfully ===")


if __name__ == "__main__":
    main()