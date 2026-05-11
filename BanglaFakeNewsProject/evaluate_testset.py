import os
import csv
from predictor import Predictor

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(BASE_DIR, "Test.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Label mapping (adjust if needed)
LABEL_MAP = {0: "Real", 1: "Fake", "0": "Real", "1": "Fake", "Real": "Real", "Fake": "Fake"}

# Load predictor
predictor = Predictor(config_path=os.path.join(BASE_DIR, "config.yaml"))

total = 0
correct = 0
results = []

with open(TEST_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row.get("image_id")
        headline = row.get("headline", "")
        description = row.get("description", "")
        label = row.get("label")
        # Compose text (headline + description)
        text = f"{headline} {description}".strip()
        # Try to resolve image path
        image_path = os.path.join(IMAGES_DIR, image_id + ".jfif")
        if not os.path.exists(image_path):
            image_path = None
        # Predict
        result = predictor.predict(text, image_path)
        pred = result["prediction"]
        gold = LABEL_MAP.get(label, label)
        is_correct = (pred == gold)
        total += 1
        correct += int(is_correct)
        results.append((image_id, pred, gold, is_correct))
        print(f"Sample {total}: Pred={pred}, Gold={gold}, {'✓' if is_correct else '✗'}")

accuracy = correct / total if total else 0
print(f"\nEvaluation complete: {correct}/{total} correct. Accuracy: {accuracy:.2%}")
