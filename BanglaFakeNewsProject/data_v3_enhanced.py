"""
Enhanced data loading for v3 model.

Key improvements:
  1. Uses BOTH headline + description (concatenated with separator)
  2. Max text length increased to 256 (from 128)
  3. Training image augmentation: RandomResizedCrop, RandomHorizontalFlip,
     ColorJitter, RandomErasing, RandAugment
  4. Validation/test: only Resize + Normalize
  5. Handles missing images gracefully
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import pandas as pd
import platform

# Constants
TEXT_MODEL_NAME = "csebuetnlp/banglabert"
MAX_LENGTH = 256  # Increased from 128 to fit headline + description
IMAGE_SIZE = 224
IMAGE_FOLDER = "images"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)


def get_train_transforms():
    """Strong augmentation for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_eval_transforms():
    """Minimal transforms for validation/test."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def find_image(image_id, image_folder):
    """Find image file by ID with multiple extension support."""
    for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.jfif']:
        path = os.path.join(image_folder, str(image_id) + ext)
        if os.path.exists(path):
            return path
    return None


class EnhancedMultimodalDataset(Dataset):
    """Enhanced dataset using headline + description with strong augmentation."""

    def __init__(self, dataframe, tokenizer, image_transform, image_folder, max_length=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_folder = image_folder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Combine headline + description for richer text
        headline = str(row.get("headline", ""))
        description = str(row.get("description", ""))
        text = headline + " </s> " + description  # Use separator token

        # Tokenize
        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Load image
        img_id = row.get("image_id", "")
        img_path = find_image(img_id, self.image_folder)
        if img_path:
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.image_transform(image)
            except Exception:
                image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        else:
            image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        label = int(row.get("label", 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


class TextOnlyDataset(Dataset):
    """Text-only dataset using headline + description."""

    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        headline = str(row.get("headline", ""))
        description = str(row.get("description", ""))
        text = headline + " </s> " + description

        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class ImageOnlyDataset(Dataset):
    """Image-only dataset."""

    def __init__(self, dataframe, image_transform, image_folder):
        self.data = dataframe.reset_index(drop=True)
        self.image_transform = image_transform
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row.get("image_id", "")
        img_path = find_image(img_id, self.image_folder)
        if img_path:
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.image_transform(image)
            except Exception:
                image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        else:
            image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        return {
            "image": image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def create_enhanced_dataloaders(base_dir, batch_size=16):
    """Create all dataloaders for enhanced training."""
    train_df = pd.read_csv(os.path.join(base_dir, "Train.csv"), low_memory=False)
    val_df = pd.read_csv(os.path.join(base_dir, "Validation.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(base_dir, "Test.csv"), low_memory=False)

    img_folder = os.path.join(base_dir, IMAGE_FOLDER)
    num_workers = 0 if platform.system() == "Windows" else 2

    # Multimodal
    train_ds = EnhancedMultimodalDataset(train_df, tokenizer, get_train_transforms(), img_folder, MAX_LENGTH)
    val_ds = EnhancedMultimodalDataset(val_df, tokenizer, get_eval_transforms(), img_folder, MAX_LENGTH)
    test_ds = EnhancedMultimodalDataset(test_df, tokenizer, get_eval_transforms(), img_folder, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Text-only
    train_text = TextOnlyDataset(train_df, tokenizer, MAX_LENGTH)
    val_text = TextOnlyDataset(val_df, tokenizer, MAX_LENGTH)
    test_text = TextOnlyDataset(test_df, tokenizer, MAX_LENGTH)

    train_text_loader = DataLoader(train_text, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_text_loader = DataLoader(val_text, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_text_loader = DataLoader(test_text, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Image-only
    train_img = ImageOnlyDataset(train_df, get_train_transforms(), img_folder)
    val_img = ImageOnlyDataset(val_df, get_eval_transforms(), img_folder)
    test_img = ImageOnlyDataset(test_df, get_eval_transforms(), img_folder)

    train_img_loader = DataLoader(train_img, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_img_loader = DataLoader(val_img, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_img_loader = DataLoader(test_img, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        "multimodal": (train_loader, val_loader, test_loader),
        "text_only": (train_text_loader, val_text_loader, test_text_loader),
        "image_only": (train_img_loader, val_img_loader, test_img_loader),
    }
