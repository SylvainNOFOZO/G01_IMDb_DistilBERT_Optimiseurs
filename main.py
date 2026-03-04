"""
main.py
-------
Point d'entrée principal du projet G01.
Orchestre l'ensemble du pipeline :
  1. Random Search (benchmark des 3 optimiseurs × 3 LR)
  2. Analyse du loss landscape + sharpness
  3. Génération des figures
  4. Affichage du bilan récapitulatif

Usage :
    python main.py
"""

import sys
import copy
import json
import torch
import numpy as np
import pandas as pd

# ─── Forçage du chemin si exécuté depuis la racine du projet ─────────────────
sys.path.insert(0, ".")

from src.data_loader   import load_imdb_subsets
from src.model_setup   import load_model_and_tokenizer, evaluate, BATCH_SIZE
from src.optimization  import run_random_search, build_optimizer, OPTIMIZER_NAMES
from src.loss_landscape import analyze_all_landscapes
from src.visualization  import generate_all_figures
from torch.utils.data   import DataLoader


# ─── Paramètres globaux ───────────────────────────────────────────────────────
N_EPOCHS    = 3
OUTPUT_DIR  = "outputs/results"
FIGURE_DIR  = "outputs/figures"
MODELS_DIR  = "outputs/models"


def main():
    print("\n" + "█" * 60)
    print("  PROJET G01 — Fine-tuning de Transformers")
    print("  Dataset  : IMDb (D01)  — Analyse de sentiments binaire")
    print("  Modèle   : DistilBERT-base-uncased (M01)")
    print("  Problém. : P01 — Benchmark d'Optimiseurs")
    print("  Méthode  : Random Search")
    print("█" * 60 + "\n")

    # ── Étape 1 : Random Search ──────────────────────────────────────────────
    results_df, best_config = run_random_search(
        output_dir = OUTPUT_DIR,
        n_epochs   = N_EPOCHS,
        batch_size = BATCH_SIZE,
    )

    # ── Étape 2 : Rechargement pour le loss landscape ────────────────────────
    # On recharge un modèle propre + les données pour l'analyse landscape
    print("\n  Chargement des modèles pour l'analyse du loss landscape…")
    ref_model, tokenizer, device = load_model_and_tokenizer()
    _, val_ds, _ = load_imdb_subsets(tokenizer, verbose=False)

    import os
    os.makedirs(MODELS_DIR, exist_ok=True)

    models_for_landscape = {}

    # Pour chaque optimiseur on réentraîne le meilleur run (best LR)
    # et on conserve le modèle final pour le landscape
    from src.model_setup   import full_training_loop
    from src.optimization  import sample_learning_rates

    for i, opt_name in enumerate(OPTIMIZER_NAMES):
        subset = results_df[results_df["optimizer"] == opt_name]
        if subset.empty:
            continue
        best_lr = float(subset.loc[subset["best_val_acc"].idxmax(),
                                   "learning_rate"])

        from src.data_loader import load_imdb_subsets as _lids
        train_ds, val_ds_fresh, _ = _lids(tokenizer, verbose=False)

        model = copy.deepcopy(ref_model).to(device)
        optimizer = build_optimizer(opt_name, model.parameters(), best_lr)

        _, best_state = full_training_loop(
            model_init    = model,
            optimizer     = optimizer,
            train_dataset = train_ds,
            val_dataset   = val_ds_fresh,
            device        = device,
            num_epochs    = N_EPOCHS,
            batch_size    = BATCH_SIZE,
            run_label     = f"Landscape — {opt_name} LR={best_lr:.2e}",
        )
        model.load_state_dict(best_state)

        # Sauvegarde du state_dict
        torch.save(best_state,
                   f"{MODELS_DIR}/{opt_name.lower()}_best.pt")

        label = f"{opt_name} | LR={best_lr:.2e}"
        models_for_landscape[label] = model

    # ── Étape 3 : Loss landscape et sharpness ────────────────────────────────
    landscape_data, sharpness_data = analyze_all_landscapes(
        models_dict = models_for_landscape,
        val_dataset = val_ds,
        device      = device,
        output_dir  = OUTPUT_DIR,
    )

    # ── Étape 4 : Figures ─────────────────────────────────────────────────────
    generate_all_figures(
        results_df     = results_df,
        landscape_data = landscape_data,
        sharpness_data = sharpness_data,
        n_epochs       = N_EPOCHS,
    )

    # ── Étape 5 : Bilan récapitulatif ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BILAN RÉCAPITULATIF")
    print("=" * 60)

    summary = results_df[["optimizer", "learning_rate",
                            "best_val_acc", "test_accuracy",
                            "test_f1", "total_time_s"]].copy()
    summary["learning_rate"] = summary["learning_rate"].apply(
        lambda x: f"{x:.2e}")
    summary = summary.sort_values("best_val_acc", ascending=False)
    summary.columns = ["Optimiseur", "LR", "Val Acc", "Test Acc",
                        "Test F1", "Temps (s)"]

    print(summary.to_string(index=False))

    print(f"\n  ★ Meilleure configuration :")
    print(f"    Optimiseur : {best_config['optimizer']}")
    print(f"    LR         : {best_config['learning_rate']:.2e}")
    print(f"    Val Acc    : {best_config['best_val_acc']:.4f}")
    print(f"    Test Acc   : {best_config['test_accuracy']:.4f}")
    print(f"    Test F1    : {best_config['test_f1']:.4f}")

    print("\n  Sharpness par optimiseur :")
    for label, sharpness in sharpness_data.items():
        print(f"    {label:40s} → {sharpness:.6f}")

    print("\n" + "=" * 60)
    print("  Pipeline terminé avec succès !")
    print(f"  Résultats  → {OUTPUT_DIR}/")
    print(f"  Figures    → {FIGURE_DIR}/")
    print(f"  Modèles    → {MODELS_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
