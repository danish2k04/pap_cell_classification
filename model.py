# model.py — Model builder and ensemble class

import torch
import torch.nn as nn
from torchvision import models

import config


def build_model(num_classes):
    """Build EfficientNetV2-S with custom classifier head."""
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


class EnsembleModel(nn.Module):
    """
    Ensemble of two EfficientNetV2-S models.
    Averages softmax probabilities from both models.
    """
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x1, x2):
        probs1 = torch.softmax(self.model1(x1), dim=1)
        probs2 = torch.softmax(self.model2(x2), dim=1)
        return (probs1 + probs2) / 2


def load_model(path, num_classes, device):
    """Load a saved model from disk."""
    model = build_model(num_classes)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    print(f'Model loaded from: {path}')
    return model


def get_criterion(class_weights, device):
    """Cross entropy loss with class weights and label smoothing."""
    return nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=config.LABEL_SMOOTH
    )


def count_parameters(model):
    """Print parameter count for a model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters:     {total:>12,}  ({total/1e6:.2f}M)')
    print(f'Trainable parameters: {trainable:>12,}  ({trainable/1e6:.2f}M)')
    return total, trainable
