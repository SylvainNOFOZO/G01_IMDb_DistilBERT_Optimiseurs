"""
visualization.py
----------------
Toutes les figures du projet G01 :
  1. Courbes de convergence (train/val loss & accuracy) par optimiseur
  2. Comparaison finale des accuracy/F1 (barres groupées)
  3. Loss landscape 1D par optimiseur
  4. Heatmap des résultats Random Search (optimiseur × learning rate)
  5. Sharpness par optimiseur (barres)

Sorties : outputs/figures/*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Thème homogène
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
CMAP      = "tab10"
FIG_DIR   = "outputs/figures"
DPI       = 150
OPTIMIZER_COLORS = {
    "AdamW"    : "#2E75B6",
    "SGD"      : "#E07B39",
    "Adafactor": "#43A047",
}


def _save(fig, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = f"{FIG_DIR}/{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure sauvegardée → {path}")


# ─── 1. Courbes de convergence ────────────────────────────────────────────────
def plot_convergence_curves(results_df: pd.DataFrame, n_epochs: int = 3):
    """
    Pour chaque optimiseur, affiche les courbes train/val loss et accuracy
    du run ayant la meilleure val_accuracy.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Courbes de convergence — Meilleur run par optimiseur\n"
        "Données : IMDb (D01)  |  Modèle : DistilBERT (M01)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    epochs = list(range(1, n_epochs + 1))

    for col, opt in enumerate(["AdamW", "SGD", "Adafactor"]):
        subset = results_df[results_df["optimizer"] == opt]
        if subset.empty:
            continue
        best_row = subset.loc[subset["best_val_acc"].idxmax()]
        color    = OPTIMIZER_COLORS.get(opt, "#666")

        # Loss
        ax_loss = axes[0, col]
        ax_loss.plot(epochs, best_row["train_loss_history"],
                     "o--", color=color, alpha=0.7, label="Train")
        ax_loss.plot(epochs, best_row["val_loss_history"],
                     "s-",  color=color, label="Val")
        ax_loss.set_title(f"{opt}\n(LR = {best_row['learning_rate']:.2e})",
                          fontsize=11)
        ax_loss.set_xlabel("Époque")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=9)
        ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Accuracy
        ax_acc = axes[1, col]
        ax_acc.plot(epochs, best_row["train_acc_history"],
                    "o--", color=color, alpha=0.7, label="Train")
        ax_acc.plot(epochs, best_row["val_acc_history"],
                    "s-",  color=color, label="Val")
        ax_acc.set_xlabel("Époque")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1)
        ax_acc.legend(fontsize=9)
        ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    _save(fig, "01_convergence_curves")


# ─── 2. Comparaison finale (barres groupées) ──────────────────────────────────
def plot_final_comparison(results_df: pd.DataFrame):
    """
    Barres groupées : best_val_acc et test_f1 pour chaque optimiseur × run.
    """
    cols_plot = ["optimizer", "learning_rate", "best_val_acc", "test_f1"]
    df_plot   = results_df[cols_plot].copy()
    df_plot["lr_label"] = df_plot["learning_rate"].apply(lambda x: f"{x:.1e}")
    df_plot["config"]   = df_plot["optimizer"] + "\n" + df_plot["lr_label"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Comparaison finale des configurations — Random Search\n"
        "IMDb | DistilBERT | Benchmark d'Optimiseurs",
        fontsize=12, fontweight="bold",
    )

    palette = {
        row["config"]: OPTIMIZER_COLORS.get(row["optimizer"], "#888")
        for _, row in df_plot.iterrows()
    }

    for ax, metric, label in zip(
        axes,
        ["best_val_acc", "test_f1"],
        ["Val Accuracy", "Test F1-Macro"],
    ):
        bars = ax.bar(
            df_plot["config"],
            df_plot[metric],
            color=[palette[c] for c in df_plot["config"]],
            edgecolor="white", linewidth=0.8, alpha=0.88,
        )
        # Annotation des valeurs
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    _save(fig, "02_final_comparison")


# ─── 3. Loss landscape 1D ────────────────────────────────────────────────────
def plot_loss_landscape(landscape_data: dict):
    """
    Trace le profil 1D du loss landscape pour chaque optimiseur sur
    la même figure (subplots).
    """
    n     = len(landscape_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Loss Landscape 1D — Perturbation des paramètres\n"
        "(direction aléatoire normalisée, filter normalization)",
        fontsize=12, fontweight="bold",
    )

    for ax, (label, (alphas, losses)) in zip(axes, landscape_data.items()):
        # Déduction de l'optimiseur depuis le label
        opt_name = label.split("|")[0].strip() if "|" in label else label
        color    = OPTIMIZER_COLORS.get(opt_name, "#555")

        ax.plot(alphas, losses, "o-", color=color, linewidth=2,
                markersize=5, label=label)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="θ optimal")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("α (perturbation)")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "03_loss_landscape")


# ─── 4. Heatmap Random Search ─────────────────────────────────────────────────
def plot_heatmap_random_search(results_df: pd.DataFrame):
    """
    Heatmap val_accuracy : lignes = optimiseurs, colonnes = learning rates.
    """
    df = results_df.copy()
    df["lr_label"] = df["learning_rate"].apply(lambda x: f"{x:.2e}")

    pivot = df.pivot_table(
        index="optimizer", columns="lr_label",
        values="best_val_acc", aggfunc="max",
    )

    fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns) * 2.2), 3.5))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f",
        cmap="YlGnBu", linewidths=0.5,
        cbar_kws={"label": "Best Val Accuracy"},
    )
    ax.set_title(
        "Heatmap Random Search — Val Accuracy\n"
        "Optimiseur × Learning Rate",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Optimiseur")
    plt.tight_layout()
    _save(fig, "04_heatmap_random_search")


# ─── 5. Sharpness ────────────────────────────────────────────────────────────
def plot_sharpness(sharpness_data: dict):
    """
    Barres horizontales de la sharpness par optimiseur.
    Un minimum plat (sharpness ≈ 0) favorise la généralisation.
    """
    labels = list(sharpness_data.keys())
    values = [sharpness_data[k] for k in labels]
    colors = [
        OPTIMIZER_COLORS.get(k.split("|")[0].strip(), "#888")
        for k in labels
    ]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(labels))))
    bars = ax.barh(labels, values, color=colors, alpha=0.85,
                   edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.5f}", va="center", fontsize=9)

    ax.set_xlabel("Sharpness  (↓ = minimum plus plat = meilleure généralisation)")
    ax.set_title(
        "Sharpness du Loss Landscape par configuration\n"
        "Sharpness = (1/N) Σ |L(θ + ε·dᵢ) − L(θ)|",
        fontsize=10, fontweight="bold",
    )
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, "05_sharpness")


# ─── Appel groupé ─────────────────────────────────────────────────────────────
def generate_all_figures(results_df, landscape_data, sharpness_data,
                          n_epochs: int = 3):
    """Lance la génération de toutes les figures."""
    print("\n" + "=" * 60)
    print("  Génération des figures")
    print("=" * 60)
    plot_convergence_curves(results_df, n_epochs=n_epochs)
    plot_final_comparison(results_df)
    plot_loss_landscape(landscape_data)
    plot_heatmap_random_search(results_df)
    plot_sharpness(sharpness_data)
    print()
