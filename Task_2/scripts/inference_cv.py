import argparse
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from models.animal_cv_classifier import AnimalClassifier


def main():
    parser = argparse.ArgumentParser(description="Inference for Computer Vision model (ResNet50)")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to Image (напр., data/test/cow.jpg)")
    parser.add_argument("--weights", type=str, default="weights/best_cv_model.pth",
                        help="Path to weights")

    args = parser.parse_args()

    print(f"=== Starting CV inference ===")
    print(f"Image: {args.image}")
    print(f"Weights: {args.weights}\n")

    try:
        model = AnimalClassifier(weights_path=args.weights)

        prediction = model.predict(args.image)
        print(f"Found animal: {prediction}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()