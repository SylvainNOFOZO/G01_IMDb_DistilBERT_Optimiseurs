"""
loss_landscape.py
-----------------
Analyse du loss landscape (méthode simplifiée 1D pour CPU).
Calcule la Sharpness de chaque optimiseur selon la formule du cours.

Sharpness = (1/N) * Σ |L(θ + ε·dᵢ) − L(θ)|

Sorties :
  • outputs/results/landscape_metrics.csv
"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


# ─── Paramètres ──────────────────────────────────────────────────────────────
N_POINTS    = 10       # Nombre de points sur l'axe α (réduit pour CPU)
EPSILON     = 0.05     # Amplitude de la perturbation
N_SAMPLES   = 100      # Exemples pour l'évaluation rapide du loss
N_DIRS      = 5        # Nombre de directions aléatoires pour moyenner la sharpness
BATCH_SIZE  = 16


# ─── Évaluation rapide sur un sous-ensemble ───────────────────────────────────
def _eval_loss_on_subset(model, dataset, n_samples: int,
                          device, batch_size: int = BATCH_SIZE) -> float:
    """
    Calcule la cross-entropy moyenne sur n_samples exemples aléatoires.
    """
    indices    = np.random.choice(len(dataset), size=min(n_samples, len(dataset)),
                                   replace=False)
    sub        = Subset(dataset, indices.tolist())
    loader     = DataLoader(sub, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    model.eval()
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
            total_loss += out.loss.item() * labels.size(0)
            total_n    += labels.size(0)

    return total_loss / total_n if total_n > 0 else float("nan")


# ─── Profil 1D du loss landscape ─────────────────────────────────────────────
def compute_loss_landscape_1d(
    model,
    dataset,
    device,
    n_points: int = N_POINTS,
    epsilon:  float = EPSILON,
    n_samples: int  = N_SAMPLES,
    seed: int = 0,
):
    """
    Perturbe les paramètres du modèle le long d'une direction aléatoire
    et mesure la variation du loss.

    Args:
        model    : modèle évalué (non modifié en sortie)
        dataset  : dataset de validation
        device   : torch.device
        n_points : nombre de points αᵢ dans [-ε, +ε]
        epsilon  : amplitude maximale de la perturbation
        n_samples: taille du sous-ensemble d'évaluation
        seed     : graine pour la direction aléatoire

    Returns:
        alphas (ndarray), losses (list of float)
    """
    torch.manual_seed(seed)
    model.eval()

    original_params = [p.clone().detach() for p in model.parameters()]

    # Direction aléatoire normalisée (filter normalization)
    direction = [torch.randn_like(p) for p in original_params]
    # Normalisation filtre par filtre pour éviter les biais de dimension
    for i, (d, p) in enumerate(zip(direction, original_params)):
        p_norm = p.norm()
        d_norm = d.norm()
        if d_norm > 0 and p_norm > 0:
            direction[i] = d * (p_norm / d_norm)

    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []

    for alpha in alphas:
        # Application de la perturbation
        with torch.no_grad():
            for p, p0, d in zip(model.parameters(), original_params, direction):
                p.data = p0 + alpha * d

        loss = _eval_loss_on_subset(model, dataset, n_samples, device)
        losses.append(loss)

    # Restauration des paramètres originaux
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), original_params):
            p.data = p0.clone()

    return alphas, losses


# ─── Calcul de la Sharpness ───────────────────────────────────────────────────
def compute_sharpness(
    model,
    dataset,
    device,
    epsilon:   float = EPSILON,
    n_dirs:    int   = N_DIRS,
    n_samples: int   = N_SAMPLES,
) -> float:
    """
    Estime la Sharpness selon la formule du cours :
        Sharpness = (1/N) * Σᵢ |L(θ + ε·dᵢ) − L(θ)|

    Plusieurs directions sont moyennées pour stabiliser l'estimation.

    Returns:
        sharpness (float)
    """
    # Loss au point de référence θ
    base_loss = _eval_loss_on_subset(model, dataset, n_samples, device)

    perturbation_losses = []
    for seed in range(n_dirs):
        _, losses = compute_loss_landscape_1d(
            model, dataset, device,
            n_points=1,
            epsilon=epsilon,
            n_samples=n_samples,
            seed=seed,
        )
        perturbation_losses.append(abs(losses[0] - base_loss))

    sharpness = float(np.mean(perturbation_losses))
    return sharpness


# ─── Analyse complète pour tous les runs ─────────────────────────────────────
def analyze_all_landscapes(
    models_dict: dict,
    val_dataset,
    device,
    output_dir: str = "outputs/results",
):
    """
    Calcule les profils 1D et la sharpness pour chaque modèle fourni.

    Args:
        models_dict : {run_label: model} avec model chargé avec best_state
        val_dataset : dataset de validation
        device      : torch.device
        output_dir  : répertoire de sauvegarde

    Returns:
        landscape_data  : {run_label: (alphas, losses)}
        sharpness_data  : {run_label: sharpness_value}
    """
    import os, pandas as pd
    os.makedirs(output_dir, exist_ok=True)

    landscape_data = {}
    sharpness_data = {}

    print("\n" + "=" * 60)
    print("  Analyse du Loss Landscape")
    print("=" * 60)

    for label, model in models_dict.items():
        print(f"\n  → {label}")

        alphas, losses = compute_loss_landscape_1d(
            model, val_dataset, device, seed=42)
        sharpness = compute_sharpness(
            model, val_dataset, device)

        landscape_data[label] = (alphas, losses)
        sharpness_data[label] = sharpness

        print(f"    Sharpness = {sharpness:.6f}")

    # Sauvegarde CSV des sharpness
    rows = [{"run_label": k, "sharpness": v}
            for k, v in sharpness_data.items()]
    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/landscape_metrics.csv", index=False)
    print(f"\n  Métriques sauvegardées → {output_dir}/landscape_metrics.csv")

    return landscape_data, sharpness_data
