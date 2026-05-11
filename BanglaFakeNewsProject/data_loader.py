"""
Data loading for Bangla Fake News Detection.

This module re-exports from preprocess_data for backward compatibility.
New code should import directly from preprocess_data.
"""

import platform
import yaml

from preprocess_data import (
    MultimodalDataset,
    prepare_datasets_from_csv,
    preprocess_image,
    train_df,
    val_df,
    test_df,
    tokenizer,
)

__all__ = [
    "MultimodalDataset",
    "create_data_loaders",
    "prepare_datasets_from_csv",
    "preprocess_image",
    "train_df",
    "val_df",
    "test_df",
    "tokenizer",
]


def create_data_loaders(config):
    """Build train/val/test DataLoaders from config. (Re-exported for backward compatibility)"""
    import pandas as pd
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from torchvision import transforms

    text_model_name = config['model']['text_model_name']
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    image_size = config['data']['image_size']

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_df_local = pd.read_csv(config['data']['train_csv'])
    val_df_local = pd.read_csv(config['data']['val_csv'])
    test_df_local = pd.read_csv(config['data']['test_csv'])

    train_dataset = MultimodalDataset(train_df_local, tokenizer, image_transform, config)
    val_dataset = MultimodalDataset(val_df_local, tokenizer, image_transform, config)
    test_dataset = MultimodalDataset(test_df_local, tokenizer, image_transform, config)

    num_workers = 0 if platform.system() == 'Windows' else config['hardware']['num_workers']
    loader_kwargs = dict(num_workers=num_workers, pin_memory=config['hardware']['pin_memory'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'], shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
