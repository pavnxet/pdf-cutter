import json, os, base64
from datetime import datetime

EXAMPLES_PATH = "data/examples/examples.json"
MAX_EXAMPLES = 10

def _load():
    if not os.path.exists(EXAMPLES_PATH):
        return []
    with open(EXAMPLES_PATH, "r") as f:
        return json.load(f)

def _save(data):
    os.makedirs(os.path.dirname(EXAMPLES_PATH), exist_ok=True)
    with open(EXAMPLES_PATH, "w") as f:
        json.dump(data, f, indent=2)

def save_correction(image_path, hindi_box, english_box):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    examples = _load()
    examples.append({
        "image_b64": img_b64,
        "boxes": {"hindi": hindi_box, "english": english_box},
        "timestamp": datetime.now().isoformat()
    })
    examples = examples[-MAX_EXAMPLES:]
    _save(examples)
    return len(examples)

def get_latest_examples(n=3):
    examples = _load()
    return examples[-n:]
