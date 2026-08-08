"""S4 training entry point. It trains the standalone RoPE implementation."""
import sys

from src.rope_model import RopeTransformerClassifier
import train


def main():
    original_classifier = train.TransformerClassifier
    train.TransformerClassifier = RopeTransformerClassifier
    try:
        train.main()
    finally:
        train.TransformerClassifier = original_classifier


if __name__ == "__main__":
    main()