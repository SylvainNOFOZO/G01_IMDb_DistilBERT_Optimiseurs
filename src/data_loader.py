"""
data_loader.py
--------------
Chargement et préparation du dataset IMDb (D01) pour le projet G01.
Inclut le sous-échantillonnage équilibré pour CPU.
"""

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
import torch


# ─── Constantes du projet ────────────────────────────────────────────────────
DATASET_NAME     = "imdb"
NUM_CLASSES      = 2
LABEL_NAMES      = ["négatif", "positif"]
MAX_SEQ_LENGTH   = 256          # Tronqué pour CPU (original : 512)
TRAIN_SAMPLES    = 1000         # Sous-échantillon par classe en entraînement
VAL_SAMPLES      = 200          # Par classe en validation
TEST_SAMPLES     = 250          # Par classe en test
RANDOM_SEED      = 42


# ─── Sous-échantillonnage équilibré ──────────────────────────────────────────
def create_balanced_subset(hf_split, n_per_class: int, seed: int = RANDOM_SEED):
    """
    Retourne un sous-ensemble équilibré avec n_per_class exemples par étiquette.

    Args:
        hf_split  : split HuggingFace (train / test)
        n_per_class: nombre d'exemples à conserver par classe
        seed      : graine aléatoire pour reproductibilité

    Returns:
        dict avec clés 'text' et 'label'
    """
    rng = np.random.default_rng(seed)
    texts, labels = [], []

    for label in range(NUM_CLASSES):
        # Indices correspondant à cette classe
        indices = [i for i, ex in enumerate(hf_split) if ex["label"] == label]
        n = min(n_per_class, len(indices))
        chosen = rng.choice(indices, size=n, replace=False)
        for idx in chosen:
            ex = hf_split[int(idx)]
            texts.append(ex["text"])
            labels.append(ex["label"])

    # Mélange global
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return {"text": list(texts), "label": list(labels)}


# ─── Dataset PyTorch ─────────────────────────────────────────────────────────
class IMDbDataset(Dataset):
    """Dataset PyTorch pour IMDb tokenisé."""

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── Fonction principale ─────────────────────────────────────────────────────
def load_imdb_subsets(tokenizer, verbose: bool = True):
    """
    Charge IMDb, sous-échantillonne, tokenise et retourne les trois splits.

    Args:
        tokenizer : tokenizer HuggingFace déjà instancié
        verbose   : affiche les statistiques de chargement

    Returns:
        train_dataset, val_dataset, test_dataset (IMDbDataset)
    """
    if verbose:
        print("=" * 60)
        print("  Chargement du dataset IMDb (D01)")
        print("=" * 60)

    raw = load_dataset(DATASET_NAME)

    # Le split 'test' original sert à la fois de validation et de test
    raw_train = raw["train"]
    raw_test  = raw["test"]

    train_data = create_balanced_subset(raw_train, TRAIN_SAMPLES, seed=RANDOM_SEED)
    val_data   = create_balanced_subset(raw_test,  VAL_SAMPLES,   seed=RANDOM_SEED + 1)
    test_data  = create_balanced_subset(raw_test,  TEST_SAMPLES,  seed=RANDOM_SEED + 2)

    if verbose:
        print(f"  Train  : {len(train_data['text'])} exemples "
              f"({TRAIN_SAMPLES}/classe)")
        print(f"  Val    : {len(val_data['text'])}  exemples "
              f"({VAL_SAMPLES}/classe)")
        print(f"  Test   : {len(test_data['text'])} exemples "
              f"({TEST_SAMPLES}/classe)")

    def tokenize(data_dict):
        return tokenizer(
            data_dict["text"],
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
        )

    train_enc = tokenize(train_data)
    val_enc   = tokenize(val_data)
    test_enc  = tokenize(test_data)

    train_dataset = IMDbDataset(train_enc, train_data["label"])
    val_dataset   = IMDbDataset(val_enc,   val_data["label"])
    test_dataset  = IMDbDataset(test_enc,  test_data["label"])

    if verbose:
        print("  Tokenisation terminée (longueur max : "
              f"{MAX_SEQ_LENGTH} tokens)")
        print("=" * 60 + "\n")

    return train_dataset, val_dataset, test_dataset
