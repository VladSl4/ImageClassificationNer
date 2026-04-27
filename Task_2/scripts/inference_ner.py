import argparse
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from models.animal_ner_model import AnimalNerModel


def main():
    parser = argparse.ArgumentParser(description="Inference for NER model (DistilBERT)")

    parser.add_argument("--text", type=str, required=True,
                        help="Text for analysis (e.g. \"I see a dog\")")
    parser.add_argument("--model_dir", type=str, default="weights/best_ner_model",
                        help="Path to model weights (e.g., weights/best_ner_model)")

    args = parser.parse_args()

    print(f"=== Starting NER inference ===")
    print(f"Text: '{args.text}'")
    print(f"Model: {args.model_dir}\n")

    try:
        model = AnimalNerModel(model_dir=args.model_dir)

        predictions = model.predict(args.text)

        if predictions:
            print(f"Found animals: {', '.join(predictions)}")
        else:
            print("Animals not found")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()