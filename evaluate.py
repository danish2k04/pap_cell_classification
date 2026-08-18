# evaluate.py — Inference, metrics, and ensemble evaluation


import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

import config


def run_inference(model, dataset, device, batch_size=16):
    """
    Windows-safe inference — no DataLoader workers.
    Loads images one by one and manually batches them.
    """
    model.eval()
    all_y, all_p = [], []
    total = len(dataset)

    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc='inference', leave=False):
            end     = min(start + batch_size, total)
            batch_x, batch_y = [], []
            for i in range(start, end):
                x, y = dataset[i]
                batch_x.append(x)
                batch_y.append(y)

            batch_x = torch.stack(batch_x).to(device)
            batch_y = torch.tensor(batch_y)

            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(batch_x)

            all_p.append(logits.argmax(1).cpu().numpy())
            all_y.append(batch_y.numpy())

    return np.concatenate(all_y), np.concatenate(all_p)


def run_ensemble_inference(ensemble, dataset_m1, dataset_m2, device, batch_size=16):
    """
    Windows-safe ensemble inference.
    Loads from both datasets and averages softmax probabilities.
    """
    ensemble.eval()
    all_y, all_probs = [], []
    total = len(dataset_m1)

    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc='ensemble', leave=False):
            end = min(start + batch_size, total)
            batch_x1, batch_x2, batch_y = [], [], []

            for i in range(start, end):
                x1, y  = dataset_m1[i]
                x2, _  = dataset_m2[i]
                batch_x1.append(x1)
                batch_x2.append(x2)
                batch_y.append(y)

            batch_x1 = torch.stack(batch_x1).to(device)
            batch_x2 = torch.stack(batch_x2).to(device)
            batch_y  = torch.tensor(batch_y)

            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                avg_probs = ensemble(batch_x1, batch_x2)

            all_probs.append(avg_probs.cpu().numpy())
            all_y.append(batch_y.numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.argmax(np.concatenate(all_probs), axis=1)
    return y_true, y_pred


def print_report(y_true, y_pred, classes, title='Report'):
    """Print classification report and confusion matrix."""
    print(f'\n{"=" * 60}')
    print(f'{title}')
    print('=' * 60)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print(f'\nMacro-F1:    {f1_score(y_true, y_pred, average="macro"):.4f}')
    print(f'Weighted-F1: {f1_score(y_true, y_pred, average="weighted"):.4f}')
    print(f'Accuracy:    {accuracy_score(y_true, y_pred):.4f}')
