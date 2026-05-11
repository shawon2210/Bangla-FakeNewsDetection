import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import os

def prepare_datasets_from_csv(csv_path, text_col='headline', image_col='image_id', label_col='label',
                              splits=(0.8, 0.1, 0.1), random_state=42):
    """Load a single CSV and split into train/val/test dataframes.

    This makes it easy to train on a single file such as the attached
    BanglaFakeNews500.csv. The function shuffles and splits the data.
    """
    # Robust CSV load: avoid dtype warnings and slow chunking on large files
    df = pd.read_csv(csv_path, low_memory=False)
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n = len(df)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)

    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)

    # Ensure required columns exist
    for col in [text_col, image_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")

    return train, val, test


# 1. Tokenizer
TEXT_MODEL_NAME = "csebuetnlp/banglabert"
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

# 2. Image Transform
IMAGE_SIZE = 224
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. Image Folder
IMAGE_FOLDER = 'images' # Assuming images are in the 'images' subdirectory

def preprocess_image(image_id, image_folder, image_transform):
    """
    Loads an image by ID, supports both .jpg and .png.
    Returns transformed tensor or None if not found.
    """
    for ext in ['.png', '.jpg', '.webp', '.jpeg', '.gif', '.jfif']:
        img_path = os.path.join(image_folder, str(image_id) + ext)
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
                return image_transform(image)
            except Exception as e:
                print(f"[WARNING] Could not open or transform image {img_path}: {e}")
                return None
    # print(f"[WARNING] Image not found for ID: {image_id}")
    return None

# 4. MultimodalDataset Class (adapted from data_loader.py)
class MultimodalDataset(Dataset):
    """Flexible multimodal dataset.

    Accepts either:
      - (dataframe) : uses module-level `tokenizer`, defaults for column names and max_length
      - (dataframe, tokenizer, text_col, image_col, label_col, max_length)

    This preserves backward compatibility with older scripts that call
    `MultimodalDataset(train_df)`.
    """
    def __init__(self, dataframe, tokenizer=None, text_col='headline', image_col='image_id', label_col='label', max_length=128):
        self.data = dataframe.reset_index(drop=True)
        # allow passing a tokenizer or use module-level tokenizer
        self.tokenizer = tokenizer if tokenizer is not None else globals().get('tokenizer')
        self.image_transform = image_transform
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.max_length = max_length
        self.image_folder = IMAGE_FOLDER

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row.get(self.image_col, "")
        text = row.get(self.text_col, "")
        label = row.get(self.label_col, 0)

        # Tokenize text with fixed padding
        encodings = self.tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)

        # Load image
        image = preprocess_image(img_id, self.image_folder, self.image_transform)
        if image is None:
            image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(int(label), dtype=torch.long)
        }


# --- Backwards-compatible module-level dataset objects ---
# Try to set `train_df`, `val_df`, `test_df` for scripts that import them.
try:
    BASE_DIR = os.path.dirname(__file__)
except Exception:
    BASE_DIR = os.getcwd()

def _local_path(name: str) -> str:
    return os.path.join(BASE_DIR, name)

train_df = val_df = test_df = None

# Preferred: explicit split files
if os.path.exists(_local_path('Train.csv')) and os.path.exists(_local_path('Validation.csv')) and os.path.exists(_local_path('Test.csv')):
    try:
        train_df = pd.read_csv(_local_path('Train.csv'), low_memory=False)
        val_df = pd.read_csv(_local_path('Validation.csv'), low_memory=False)
        test_df = pd.read_csv(_local_path('Test.csv'), low_memory=False)
    except Exception as e:
        print(f"[WARNING] Failed to read Train/Validation/Test CSVs: {e}")

# Fallback: single merged CSV to split
if train_df is None and os.path.exists(_local_path('merged_data.csv')):
    try:
        train_df, val_df, test_df = prepare_datasets_from_csv(_local_path('merged_data.csv'))
    except Exception as e:
        print(f"[WARNING] Failed to split merged_data.csv: {e}")

# Final fallback: try generic Train/Validation/Test lowercase
if train_df is None and os.path.exists(_local_path('train.csv')):
    try:
        train_df = pd.read_csv(_local_path('train.csv'), low_memory=False)
        val_df = pd.read_csv(_local_path('validation.csv'), low_memory=False) if os.path.exists(_local_path('validation.csv')) else pd.DataFrame(columns=["headline","image_id","label"])
        test_df = pd.read_csv(_local_path('test.csv'), low_memory=False) if os.path.exists(_local_path('test.csv')) else pd.DataFrame(columns=["headline","image_id","label"])
    except Exception as e:
        print(f"[WARNING] Failed to read lowercase train/validation/test CSVs: {e}")

if train_df is None:
    # As a last resort, create empty dataframes with expected columns so imports don't fail.
    print("[INFO] No Train/Validation/Test/merged_data CSVs found. Creating empty placeholders for train_df, val_df, test_df.")
    train_df = pd.DataFrame(columns=['headline', 'image_id', 'label'])
    val_df = pd.DataFrame(columns=['headline', 'image_id', 'label'])
    test_df = pd.DataFrame(columns=['headline', 'image_id', 'label'])
