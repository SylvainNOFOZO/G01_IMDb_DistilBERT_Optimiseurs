"""
model_setup.py
--------------
Initialisation de DistilBERT (M01) pour la classification binaire IMDb.
Fournit des utilitaires d'entraînement adaptés aux contraintes CPU.
"""

import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data_loader import NUM_CLASSES


# ─── Constantes ──────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
BATCH_SIZE   = 16        # Adapté CPU ; réduire à 8 si OOM
NUM_EPOCHS   = 3
WEIGHT_DECAY = 1e-4      # Fixé pour ce benchmark (seul l'optimiseur varie)
WARMUP_STEPS = 50


# ─── Chargement ──────────────────────────────────────────────────────────────
def get_device():
    """Sélectionne automatiquement CUDA si disponible, sinon CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositif : {device}")
    return device


def load_model_and_tokenizer(num_labels: int = NUM_CLASSES):
    """
    Charge DistilBERT-base-uncased et son tokenizer.

    Returns:
        model, tokenizer, device
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        torch_dtype=torch.float32,   # float32 obligatoire sur CPU
    )
    model = model.to(device)

    # Optimisations CPU PyTorch
    if device.type == "cpu":
        torch.set_num_threads(4)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Modèle : {MODEL_NAME}  |  Paramètres entraînables : "
          f"{n_params:,}")

    return model, tokenizer, device


# ─── Une époque d'entraînement ────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    """
    Effectue une passe d'entraînement complète.

    Returns:
        avg_loss (float), accuracy (float)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Train", leave=False, ncols=80):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds       = outputs.logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ─── Évaluation ──────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """
    Évalue le modèle sur un DataLoader.

    Returns:
        avg_loss, accuracy, f1_macro
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval ", leave=False, ncols=80):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            total_loss += outputs.loss.item() * labels.size(0)
            preds       = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n        = len(all_labels)
    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


# ─── Boucle d'entraînement complète ──────────────────────────────────────────
def full_training_loop(
    model_init,
    optimizer,
    train_dataset,
    val_dataset,
    device,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    run_label: str  = "run",
):
    """
    Entraîne un modèle frais (copie profonde de model_init) pendant num_epochs.

    Args:
        model_init   : modèle de référence (ses poids seront copiés)
        optimizer    : instance d'optimiseur déjà construite sur le modèle courant
        train_dataset: IMDbDataset d'entraînement
        val_dataset  : IMDbDataset de validation
        device       : torch.device
        num_epochs   : nombre d'époques
        batch_size   : taille des mini-batches
        run_label    : identifiant affiché dans les logs

    Returns:
        history (dict) avec listes 'train_loss', 'val_loss',
                                   'train_acc',  'val_acc',
                                   'val_f1', 'epoch_times'
        best_model_state (state_dict du meilleur modèle sur val_acc)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "val_f1":     [], "epoch_times": [],
    }

    best_val_acc       = 0.0
    best_model_state   = None

    print(f"\n{'─'*60}")
    print(f"  Run : {run_label}")
    print(f"{'─'*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model_init, train_loader, optimizer, device)
        va_loss, va_acc, va_f1 = evaluate(
            model_init, val_loader,   device)

        elapsed = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["epoch_times"].append(elapsed)

        print(f"  Époque {epoch}/{num_epochs} | "
              f"Train loss={tr_loss:.4f}  acc={tr_acc:.3f} | "
              f"Val loss={va_loss:.4f}  acc={va_acc:.3f}  "
              f"F1={va_f1:.3f} | {elapsed:.0f}s")

        if va_acc > best_val_acc:
            best_val_acc     = va_acc
            best_model_state = copy.deepcopy(model_init.state_dict())

    return history, best_model_state
