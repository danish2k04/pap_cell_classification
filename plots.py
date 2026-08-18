import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

import config

# results folder 
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# Helpers
def _save(fig, filename):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# Training History Tracking

class TrainingHistory:
    """
    Accumulates per-epoch metrics during training.
    Pass an instance to train_model and call .update() each epoch.
    """
    def __init__(self, model_name='Model'):
        self.model_name   = model_name
        self.train_loss   = []
        self.val_loss     = []
        self.val_macro_f1 = []
        self.val_weighted_f1 = []
        self.val_acc      = []

    def update(self, train_loss, val_loss, val_acc, val_macro_f1, val_weighted_f1):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.val_macro_f1.append(val_macro_f1)
        self.val_weighted_f1.append(val_weighted_f1)


# Plot 1 — Loss Curve
def plot_loss(history1, history2=None):
    """
    Plot train vs val loss for one or two models.
    history1 and history2 are TrainingHistory instances.
    """
    fig, axes = plt.subplots(
        1, 2 if history2 else 1,
        figsize=(14 if history2 else 7, 5)
    )
    if history2 is None:
        axes = [axes]

    for ax, history in zip(axes, [h for h in [history1, history2] if h]):
        epochs = range(1, len(history.train_loss) + 1)
        ax.plot(epochs, history.train_loss, 'b-o', markersize=4, label='Train Loss')
        ax.plot(epochs, history.val_loss,   'r-o', markersize=4, label='Val Loss')
        ax.set_title(f'{history.model_name} — Loss Curve', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle('Training vs Validation Loss', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'loss_curves.png')


# Plot 2 — F1 Score Curve
def plot_f1(history1, history2=None):
    """
    Plot macro-F1 and weighted-F1 per epoch for one or two models.
    """
    fig, axes = plt.subplots(
        1, 2 if history2 else 1,
        figsize=(14 if history2 else 7, 5)
    )
    if history2 is None:
        axes = [axes]

    for ax, history in zip(axes, [h for h in [history1, history2] if h]):
        epochs = range(1, len(history.val_macro_f1) + 1)
        ax.plot(
        epochs,
        history.val_macro_f1,
        marker='o',
        linestyle='-',
        color='green',
        markersize=4,
        label='Val Macro-F1'
        )

        ax.plot(
            epochs,
            history.val_weighted_f1,
            marker='p',
            linestyle='-',
            color='purple',
            markersize=4,
            label='Val Weighted-F1'
        )

        ax.plot(
            epochs,
            history.val_acc,
            marker='o',
            linestyle='--',
            color='blue',
            markersize=3,
            alpha=0.6,
            label='Val Accuracy'
        )
        ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Target (0.85)')
        ax.set_title(f'{history.model_name} — F1 Score Curve', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle('Validation F1 Scores per Epoch', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'f1_curves.png')


# Plot 3 — Confusion Matrix

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', filename='confusion_matrix.png'):
    """
    Plot a styled confusion matrix heatmap.
    """
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=axes[0], linewidths=0.5
    )
    axes[0].set_title(f'{title}\n(Raw Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # percentage
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=axes[1], linewidths=0.5
    )
    axes[1].set_title(f'{title}\n(% per True Class)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    fig.tight_layout()
    _save(fig, filename)



# Plot 4 — All Confusion Matrices 

def plot_all_confusion_matrices(
    y_true_m1, y_pred_m1,
    y_true_m2, y_pred_m2,
    y_true_ens, y_pred_ens,
    classes
):
    """
    Plot confusion matrices for Model 1, Model 2, and Ensemble side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles    = ['Model 1', 'Model 2', 'Ensemble']
    pairs     = [
        (y_true_m1, y_pred_m1),
        (y_true_m2, y_pred_m2),
        (y_true_ens, y_pred_ens),
    ]

    for ax, (y_true, y_pred), title in zip(axes, pairs, titles):
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(
            cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            ax=ax, linewidths=0.5
        )
        f1 = cm.diagonal().sum() / cm.sum()
        ax.set_title(f'{title}\n(% per True Class)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    fig.suptitle('Confusion Matrices — Model 1 vs Model 2 vs Ensemble',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'all_confusion_matrices.png')



# Plot 5 — ROC AUC Curves (One vs Rest, multiclass)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_roc_auc(model, dataset, classes, device, title='ROC AUC', filename='roc_auc.png', batch_size=16):
    """
    Plot ROC curves for each class (One vs Rest) and compute AUC.
    Works with any model that returns logits.
    """
    import torch
    model.eval()
    all_probs, all_y = [], []
    total = len(dataset)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end     = min(start + batch_size, total)
            batch_x, batch_y = [], []
            for i in range(start, end):
                x, y = dataset[i]
                batch_x.append(x)
                batch_y.append(y)
            batch_x = torch.stack(batch_x).to(device)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_y.append(np.array(batch_y))

    y_true  = np.concatenate(all_y)
    y_probs = np.concatenate(all_probs)

    # binarize labels for OvR
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ['#2196F3', '#4CAF50', '#F44336']

    auc_scores = {}
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{cls} (AUC = {roc_auc:.4f})')

    # macro average AUC
    macro_auc = np.mean(list(auc_scores.values()))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{title}\nMacro-Average AUC = {macro_auc:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, filename)
    print(f'AUC scores: {auc_scores}')
    print(f'Macro-average AUC: {macro_auc:.4f}')
    return auc_scores, macro_auc


def plot_roc_auc_ensemble(ensemble, dataset_m1, dataset_m2, classes, device,
                           title='Ensemble ROC AUC', filename='roc_auc_ensemble.png',
                           batch_size=16):
    """
    Plot ROC AUC curves for the ensemble model.
    """
    import torch
    ensemble.eval()
    all_probs, all_y = [], []
    total = len(dataset_m1)

    with torch.no_grad():
        for start in range(0, total, batch_size):
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
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                avg_probs = ensemble(batch_x1, batch_x2)
            all_probs.append(avg_probs.cpu().numpy())
            all_y.append(np.array(batch_y))

    y_true  = np.concatenate(all_y)
    y_probs = np.concatenate(all_probs)
    y_bin   = label_binarize(y_true, classes=list(range(len(classes))))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ['#2196F3', '#4CAF50', '#F44336']

    auc_scores = {}
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{cls} (AUC = {roc_auc:.4f})')

    macro_auc = np.mean(list(auc_scores.values()))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{title}\nMacro-Average AUC = {macro_auc:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, filename)
    print(f'Ensemble AUC scores: {auc_scores}')
    print(f'Ensemble Macro-average AUC: {macro_auc:.4f}')
    return auc_scores, macro_auc