import os
import json
import time
import platform
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import timm



DATA_DIR   = "data"
OUT_DIR    = "artifacts"
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 3e-4
VAL_RATIO  = 0.2
SEED       = 42

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_WORKERS = 0 if platform.system().lower().startswith("win") else 4


tfm_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
tfm_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])



def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_confusion_matrix(cm: np.ndarray, classes, out_path: str, title: str = "Confusion Matrix"):
    fig = plt.figure(figsize=(5, 4), dpi=140)
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            plt.text(j, i, str(val),
                     ha="center", va="center",
                     color="white" if val > thresh else "black",
                     fontsize=9)
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)

    full_ds = datasets.ImageFolder(DATA_DIR, transform=tfm_train)
    classes = full_ds.classes
    num_classes = len(classes)

    targets = np.array(full_ds.targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_ds = Subset(full_ds, train_idx)

    val_base = datasets.ImageFolder(DATA_DIR, transform=tfm_eval)
    val_ds   = Subset(val_base, val_idx)

    train_targets = targets[train_idx]
    counts = Counter(train_targets)
    print("Classes:", classes)
    print("Train counts:", {classes[i]: int(counts[i]) for i in range(num_classes)})

    freq = np.array([counts[i] for i in range(num_classes)], dtype=np.float32)
    class_weights = (freq.sum() / (freq * num_classes))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )


    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Mixed precision (novi API)
    from torch.amp import GradScaler, autocast
    scaler = GradScaler(device='cuda', enabled=(DEVICE == "cuda"))

    best_acc = 0.0
    best_path = os.path.join(OUT_DIR, "model_best.pth")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * y.size(0)
            train_correct  += (logits.argmax(1) == y).sum().item()
            train_total    += y.size(0)

        train_loss = train_loss_sum / max(1, train_total)
        train_acc  = train_correct / max(1, train_total)


        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_trues.extend(y.cpu().numpy().tolist())

        val_acc = val_correct / max(1, val_total)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss {train_loss:.4f} | acc {train_acc:.3f} | val_acc {val_acc:.3f} | {dt:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ saved best to {best_path} (val_acc={best_acc:.3f})")


    with open(os.path.join(OUT_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print("Best val_acc:", best_acc)

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_trues.extend(y.cpu().numpy().tolist())

    report_txt = classification_report(all_trues, all_preds, target_names=classes, digits=4)
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)
    print("\n" + report_txt)

    cm = confusion_matrix(all_trues, all_preds, labels=list(range(num_classes)))
    save_confusion_matrix(cm, classes, os.path.join(OUT_DIR, "confusion_matrix.png"),
                          title="Confusion Matrix (val)")

    print(f"Saved artifacts in: {OUT_DIR}")


if __name__ == "__main__":
    main()
