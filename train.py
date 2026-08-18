import os
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

import config


def run_epoch(model, loader, train, device, optimizer=None, scaler=None, criterion=None):
    """
    Run one epoch of training or validation.
    Returns: (loss, accuracy, macro_f1, weighted_f1, y_true, y_pred)
    """
    model.train(train)
    all_y, all_p, total_loss = [], [], 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for x, y in tqdm(loader, desc='train' if train else 'val  ', leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(x)
                loss   = criterion(logits, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            all_y.append(y.cpu().numpy())
            all_p.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    return (
        total_loss / len(y_true),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='weighted'),
        y_true,
        y_pred,
    )


def train_model(model, train_loader, val_loader, criterion,
                ckpt_path, best_path, device,
                model_name='Model', history=None,
                pretrained_path=None):          

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_state, best_weighted_f1 = None, -1.0    
    start_epoch, patience_counter = 1, 0

    if os.path.exists(ckpt_path):
        print(f'[{model_name}] Resuming from checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        best_state       = ckpt['best_state']
        best_weighted_f1    = ckpt['best_weighted_f1']
        start_epoch      = ckpt['epoch'] + 1
        patience_counter = ckpt['patience_counter']
        if history and 'history' in ckpt:
            history.train_loss      = ckpt['history']['train_loss']
            history.val_loss        = ckpt['history']['val_loss']
            history.val_acc         = ckpt['history']['val_acc']
            history.val_macro_f1    = ckpt['history']['val_macro_f1']
            history.val_weighted_f1 = ckpt['history']['val_weighted_f1']
        print(f'[{model_name}] Resuming from epoch {start_epoch} | '
              f'best weighted-F1: {best_weighted_f1:.4f}')
    else:
        # load pretrained pap cell weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            model.load_state_dict(
                torch.load(pretrained_path, map_location=device, weights_only=False)
            )
            print(f'[{model_name}] Loaded pretrained weights from {pretrained_path}')
        else:
            print(f'[{model_name}] Starting fresh from ImageNet weights.')

    for epoch in range(start_epoch, config.EPOCHS + 1):
        tr = run_epoch(model, train_loader, train=True,
                       device=device, optimizer=optimizer,
                       scaler=scaler, criterion=criterion)
        va = run_epoch(model, val_loader, train=False,
                       device=device, criterion=criterion)
        scheduler.step()

        if history:
            history.update(
                train_loss=tr[0], val_loss=va[0],
                val_acc=va[1], val_macro_f1=va[2],
                val_weighted_f1=va[3]
            )

        print(f'[{model_name}] Epoch {epoch:02d} | '
              f'train_loss={tr[0]:.4f} | val_loss={va[0]:.4f} | '
              f'val_acc={va[1]:.4f} | val_macro_f1={va[2]:.4f} | '
              f'val_weighted_f1={va[3]:.4f}')

        # save on best weighted-F1
        if va[3] > best_weighted_f1:
            best_weighted_f1    = va[3]
            best_state       = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            torch.save(best_state, best_path)
            print(f'  ✓ [{model_name}] New best macro-F1: {best_weighted_f1:.4f} — saved.')
        else:
            patience_counter += 1

        torch.save({
            'epoch':            epoch,
            'model_state':      model.state_dict(),
            'optimizer_state':  optimizer.state_dict(),
            'scheduler_state':  scheduler.state_dict(),
            'scaler_state':     scaler.state_dict(),
            'best_state':       best_state,
            'best_weighted_f1':    best_weighted_f1,    
            'patience_counter': patience_counter,
            'history': {
                'train_loss':      history.train_loss      if history else [],
                'val_loss':        history.val_loss        if history else [],
                'val_acc':         history.val_acc         if history else [],
                'val_macro_f1':    history.val_macro_f1    if history else [],
                'val_weighted_f1': history.val_weighted_f1 if history else [],
            }
        }, ckpt_path)

        if patience_counter >= config.PATIENCE:
            print(f'[{model_name}] Early stopping at epoch {epoch}.')
            break

    print(f'[{model_name}] Training done. Best weighted-F1: {best_weighted_f1:.4f}')