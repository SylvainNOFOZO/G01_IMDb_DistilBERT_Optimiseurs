"""
optimization.py
---------------
Problématique P01 – Benchmark d'Optimiseurs avec Random Search.

Protocole :
  • 3 optimiseurs : AdamW, SGD (avec momentum), Adafactor
  • 3 learning rates tirés aléatoirement dans [1e-6, 5e-4] (log-uniforme)
  • Tous les autres hyperparamètres sont fixés
  • Chaque configuration est évaluée → meilleure val_accuracy sélectionnée

Sorties :
  • outputs/results/random_search_results.csv
  • outputs/results/best_config.json
"""

import json
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.model_setup import (
    load_model_and_tokenizer,
    full_training_loop,
    evaluate,
    BATCH_SIZE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)
from src.data_loader import load_imdb_subsets
from torch.utils.data import DataLoader

# ─── Tentative d'import Adafactor (transformers) ────────────────────────────
try:
    from transformers.optimization import Adafactor
    HAS_ADAFACTOR = True
except ImportError:
    HAS_ADAFACTOR = False
    print("  [WARN] Adafactor non disponible ; remplacé par RMSprop.")


# ─── Espace de recherche ─────────────────────────────────────────────────────
OPTIMIZER_NAMES = ["AdamW", "SGD", "Adafactor"]
N_LR_PER_OPT   = 3           # 3 learning rates par optimiseur
LR_LOW          = 1e-6
LR_HIGH         = 5e-4
RANDOM_SEED     = 42


# ─── Tirage log-uniforme des learning rates ───────────────────────────────────
def sample_learning_rates(n: int, low: float = LR_LOW,
                           high: float = LR_HIGH,
                           seed: int = RANDOM_SEED) -> list:
    """
    Tire n valeurs log-uniformes dans [low, high].
    La distribution log-uniforme est standard pour les learning rates.
    """
    rng = np.random.default_rng(seed)
    log_low  = np.log(low)
    log_high = np.log(high)
    return sorted(rng.uniform(log_low, log_high, size=n).tolist())


# ─── Construction de l'optimiseur ────────────────────────────────────────────
def build_optimizer(name: str, model_params, lr: float,
                    weight_decay: float = WEIGHT_DECAY):
    """
    Instancie l'optimiseur demandé avec les paramètres donnés.

    Args:
        name         : 'AdamW', 'SGD', ou 'Adafactor'
        model_params : model.parameters()
        lr           : learning rate
        weight_decay : régularisation L2

    Returns:
        torch.optim.Optimizer
    """
    if name == "AdamW":
        return optim.AdamW(model_params, lr=lr,
                           weight_decay=weight_decay,
                           eps=1e-8)

    elif name == "SGD":
        return optim.SGD(model_params, lr=lr,
                         momentum=0.9,
                         weight_decay=weight_decay,
                         nesterov=True)

    elif name == "Adafactor":
        if HAS_ADAFACTOR:
            # Adafactor ne prend pas de lr explicite en mode adaptatif
            # on force scale_parameter=False pour imposer notre lr
            return Adafactor(
                model_params,
                lr=lr,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=weight_decay,
            )
        else:
            # Repli sur RMSprop si Adafactor absent
            return optim.RMSprop(model_params, lr=lr,
                                  weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimiseur inconnu : {name}")


# ─── Random Search principal ─────────────────────────────────────────────────
def run_random_search(
    output_dir: str = "outputs/results",
    n_epochs:   int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """
    Exécute le benchmark Random Search sur les 3 optimiseurs x 3 LR.

    Args:
        output_dir : répertoire de sortie pour les résultats CSV/JSON
        n_epochs   : nombre d'époques par run
        batch_size : taille de batch

    Returns:
        results_df (DataFrame pandas), best_config (dict)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Chargement modèle de référence et tokenizer
    print("\n" + "=" * 60)
    print("  G01 — Benchmark d'Optimiseurs (P01) — Random Search")
    print("=" * 60)

    ref_model, tokenizer, device = load_model_and_tokenizer()

    # 2. Chargement des données
    train_ds, val_ds, test_ds = load_imdb_subsets(tokenizer)

    # 3. Tirage des learning rates
    lr_samples = {
        opt_name: [np.exp(v) for v in sample_learning_rates(
            N_LR_PER_OPT, seed=RANDOM_SEED + i)]
        for i, opt_name in enumerate(OPTIMIZER_NAMES)
    }

    print("\n  Espace de recherche :")
    for opt, lrs in lr_samples.items():
        print(f"    {opt:12s} → LR = {[f'{v:.2e}' for v in lrs]}")
    print()

    # 4. Boucle de recherche
    records = []

    for opt_name in OPTIMIZER_NAMES:
        for run_idx, lr in enumerate(lr_samples[opt_name]):

            run_label = f"{opt_name} | LR={lr:.2e}"

            # Réinitialisation propre du modèle à chaque run
            model = copy.deepcopy(ref_model)
            model.to(device)

            optimizer = build_optimizer(opt_name, model.parameters(),
                                        lr=lr)

            history, best_state = full_training_loop(
                model_init   = model,
                optimizer    = optimizer,
                train_dataset= train_ds,
                val_dataset  = val_ds,
                device       = device,
                num_epochs   = n_epochs,
                batch_size   = batch_size,
                run_label    = run_label,
            )

            # Évaluation finale sur le test set avec le meilleur état
            model.load_state_dict(best_state)
            test_loader = DataLoader(test_ds, batch_size=batch_size,
                                     shuffle=False, num_workers=0)
            _, test_acc, test_f1 = evaluate(model, test_loader, device)

            record = {
                "optimizer"   : opt_name,
                "learning_rate": lr,
                "run_idx"     : run_idx + 1,
                # Meilleures valeurs de validation sur toutes les époques
                "best_val_acc": max(history["val_acc"]),
                "best_val_f1" : max(history["val_f1"]),
                "final_val_loss": history["val_loss"][-1],
                "test_accuracy": test_acc,
                "test_f1"     : test_f1,
                # Temps total d'entraînement
                "total_time_s": sum(history["epoch_times"]),
                # Historique complet sérialisé
                "train_loss_history": history["train_loss"],
                "val_loss_history"  : history["val_loss"],
                "train_acc_history" : history["train_acc"],
                "val_acc_history"   : history["val_acc"],
            }
            records.append(record)

            print(f"\n  ✓ {run_label} → "
                  f"val_acc={record['best_val_acc']:.4f}  "
                  f"test_acc={test_acc:.4f}  "
                  f"F1={test_f1:.4f}")

    # 5. Construction du DataFrame
    results_df = pd.DataFrame(records)

    # 6. Sauvegarde CSV (sans les listes d'historique)
    cols_csv = ["optimizer", "learning_rate", "run_idx",
                "best_val_acc", "best_val_f1", "final_val_loss",
                "test_accuracy", "test_f1", "total_time_s"]
    results_df[cols_csv].to_csv(
        f"{output_dir}/random_search_results.csv", index=False)
    print(f"\n  Résultats sauvegardés → {output_dir}/random_search_results.csv")

    # 7. Meilleure configuration
    best_row   = results_df.loc[results_df["best_val_acc"].idxmax()]
    best_config = {
        "optimizer"    : best_row["optimizer"],
        "learning_rate": float(best_row["learning_rate"]),
        "best_val_acc" : float(best_row["best_val_acc"]),
        "test_accuracy": float(best_row["test_accuracy"]),
        "test_f1"      : float(best_row["test_f1"]),
    }
    with open(f"{output_dir}/best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  MEILLEURE CONFIGURATION")
    print(f"{'=' * 60}")
    for k, v in best_config.items():
        print(f"    {k:20s} : {v}")
    print(f"{'=' * 60}\n")

    return results_df, best_config
