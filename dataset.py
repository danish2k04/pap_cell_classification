# dataset.py — Dataset classes and transforms


import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms

import config


# Transforms

class PadToSquareWhite:
    """Pads image to square with white background before resizing."""
    def __call__(self, img):
        w, h = img.size
        m    = max(w, h)
        l    = (m - w) // 2
        t    = (m - h) // 2
        return ImageOps.expand(img, border=(l, t, m - w - l, m - h - t), fill=(255, 255, 255))


def get_train_tf(img_size):
    """Standard augmentation transform for Model 1."""
    return transforms.Compose([
        PadToSquareWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20, fill=255),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_heavy_tf(img_size):
    """Heavy augmentation transform for unhealthy class in Model 2."""
    return transforms.Compose([
        PadToSquareWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10, fill=255),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_standard_tf_m2(img_size):
    """Standard augmentation for healthy/rubbish in Model 2."""
    return transforms.Compose([
        PadToSquareWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_eval_tf(img_size):
    """Evaluation transform — no augmentation."""
    return transforms.Compose([
        PadToSquareWhite(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])



# Dataset Classes

class PapCellDataset(Dataset):
    """Training/validation dataset — images stored in label subfolders."""
    def __init__(self, df, img_root, tfm):
        self.df       = df.reset_index(drop=True)
        self.img_root = img_root
        self.tfm      = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['label'], row['image_name'])
        img      = Image.open(img_path).convert('RGB')
        return self.tfm(img), int(row['target'])


class PapCellDatasetDualAug(Dataset):
    """Model 2 dataset — heavy augmentation for unhealthy, standard for others."""
    def __init__(self, df, img_root, tfm, heavy_tfm, unhealthy_idx):
        self.df            = df.reset_index(drop=True)
        self.img_root      = img_root
        self.tfm           = tfm
        self.heavy_tfm     = heavy_tfm
        self.unhealthy_idx = unhealthy_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['label'], row['image_name'])
        img      = Image.open(img_path).convert('RGB')
        if int(row['target']) == self.unhealthy_idx:
            return self.heavy_tfm(img), int(row['target'])
        return self.tfm(img), int(row['target'])


class PapCellTestDataset(Dataset):
    """Test dataset — flat folder, no label subfolders."""
    def __init__(self, df, img_root, tfm):
        self.df       = df.reset_index(drop=True)
        self.img_root = img_root
        self.tfm      = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['image_name'])
        img      = Image.open(img_path).convert('RGB')
        return self.tfm(img), int(row['target'])



# Data Loading Functions

def load_dataframe():
    """Load training CSV, drop bothcells, build label map."""
    df = pd.read_csv(config.TRAIN_CSV)
    if config.DROP_BOTHCELLS:
        df = df[df['label'] != 'bothcells'].copy()

    classes      = sorted(df['label'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df['target'] = df['label'].map(class_to_idx)

    print(f'Classes: {classes}')
    print(f'Label distribution:\n{df["label"].value_counts()}')
    return df, classes, class_to_idx


def split_data(df):
    """Stratified train/val split."""
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config.VAL_SIZE, random_state=config.SEED
    )
    train_idx, val_idx = next(sss.split(df['image_name'], df['target']))
    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()
    print(f'Train: {len(df_train)} | Val: {len(df_val)}')
    return df_train, df_val


def get_class_weights(df_train):
    """Compute inverse frequency class weights."""
    counts        = df_train['target'].value_counts().sort_index().values.astype('float32')
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.mean()
    return class_weights


def get_sampler(df_train, class_weights):
    """Build weighted random sampler."""
    sample_weights = df_train['target'].map(
        {i: w for i, w in enumerate(class_weights)}
    ).values
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def get_loaders_m1(df_train, df_val, sampler):
    """DataLoaders for Model 1 — standard augmentation at IMG_SIZE_M1."""
    train_ds = PapCellDataset(df_train, config.TRAIN_DIR, get_train_tf(config.IMG_SIZE_M1))
    val_ds   = PapCellDataset(df_val,   config.TRAIN_DIR, get_eval_tf(config.IMG_SIZE_M1))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, val_loader


def get_loaders_m2(df_train, df_val, sampler, unhealthy_idx):
    """DataLoaders for Model 2 — dual augmentation at IMG_SIZE_M2."""
    train_ds = PapCellDatasetDualAug(
        df_train, config.TRAIN_DIR,
        tfm=get_standard_tf_m2(config.IMG_SIZE_M2),
        heavy_tfm=get_heavy_tf(config.IMG_SIZE_M2),
        unhealthy_idx=unhealthy_idx
    )
    val_ds = PapCellDataset(df_val, config.TRAIN_DIR, get_eval_tf(config.IMG_SIZE_M2))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, val_loader


def get_test_loaders(df_test):
    """Test DataLoaders — one per model size."""
    test_ds_m1 = PapCellTestDataset(df_test, config.TEST_DATA_DIR, get_eval_tf(config.IMG_SIZE_M1))
    test_ds_m2 = PapCellTestDataset(df_test, config.TEST_DATA_DIR, get_eval_tf(config.IMG_SIZE_M2))

    test_loader_m1 = DataLoader(test_ds_m1, batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=2, pin_memory=False)
    test_loader_m2 = DataLoader(test_ds_m2, batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=2, pin_memory=False)
    return test_loader_m1, test_loader_m2, test_ds_m1, test_ds_m2


def load_test_df(class_to_idx):
    """Load test CSV, drop bothcells, map labels."""
    TEST_CSV_PATH = None
    for p in config.TEST_CSV_CANDIDATES:
        if os.path.exists(p):
            TEST_CSV_PATH = p
            print(f'Found test CSV: {p}')
            break

    if TEST_CSV_PATH is None:
        raise FileNotFoundError('No test CSV found. Check TEST_CSV_CANDIDATES in config.py')

    df_test = pd.read_csv(TEST_CSV_PATH)
    if config.DROP_BOTHCELLS:
        df_test = df_test[df_test['label'] != 'bothcells'].copy()
    df_test['target'] = df_test['label'].map(class_to_idx)

    print(f'Test samples: {len(df_test)}')
    print(df_test['label'].value_counts())
    return df_test
