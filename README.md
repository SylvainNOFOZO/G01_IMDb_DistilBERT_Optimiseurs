# G01 — Fine-tuning de Transformers : Benchmark d'Optimiseurs

**Projet** : Fine-tuning de Transformers — Optimisation d'hyperparamètres  
**Groupe** : G01  
**Cours** : Optimisation & Loss Landscape  
**Enseignante** : MBIA NDI Marie Thérèse  
**Date limite** : 13 mars 2026  

---

## Attribution

| Critère | Valeur |
|---------|--------|
| Dataset | D01 — IMDb reviews (50k, Anglais, 2 classes) |
| Modèle | M01 — DistilBERT-base-uncased (66M paramètres) |
| Problématique | P01 — Benchmark d'Optimiseurs |
| Méthode | Random Search |

---

## Problématique (P01)

> **Quel optimiseur (AdamW, SGD avec momentum, Adafactor) donne les meilleurs résultats pour le fine-tuning de DistilBERT sur IMDb ?**

**Protocole :**
- Fixer tous les autres hyperparamètres (weight_decay=1e-4, batch_size=16, epochs=3)
- 3 learning rates tirés aléatoirement dans [1e-6, 5e-4] (log-uniforme) par optimiseur
- Évaluation : Val Accuracy + Test F1-Macro
- Analyse du loss landscape pour chaque meilleur modèle
- Calcul de la Sharpness comme métrique de platitude du minima

---

## Structure du projet

```
G01_IMDb_DistilBERT_Optimiseurs/
├── main.py                    ← Pipeline complet (entraînement + figures)
├── dashboard.py               ← Dashboard interactif Dash/Plotly
├── pyproject.toml             ← Configuration Poetry (dépendances + métadonnées)
├── poetry.lock                ← Versions exactes figées (généré par Poetry)
├── requirements.txt           ← Fallback pip (si Poetry non disponible)
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py         ← Chargement & sous-échantillonnage IMDb
│   ├── model_setup.py         ← DistilBERT + boucle d'entraînement
│   ├── optimization.py        ← Random Search sur 3 optimiseurs × 3 LR
│   ├── loss_landscape.py      ← Profil 1D & calcul de Sharpness
│   └── visualization.py       ← Figures matplotlib/seaborn statiques
├── notebooks/
│   ├── exploration.ipynb      ← Exploration du dataset IMDb
│   └── analysis.ipynb         ← Analyse des résultats post-entraînement
├── outputs/
│   ├── results/               ← CSV et JSON des résultats
│   ├── figures/               ← Figures PNG statiques
│   └── models/                ← State dicts des meilleurs modèles
└── data/                      ← (optionnel) cache local
```

---

## Installation

### Option A — Avec Poetry (recommandé)

Poetry garantit la reproductibilité exacte de l'environnement grâce au `poetry.lock`.

```bash
# 1. Installer Poetry (une seule fois)
pip install poetry

# 2. Cloner / aller dans le dossier du projet
cd "C:\Users\SYLVAIN\Desktop\Projet ML OP\G01_IMDb_DistilBERT_Optimiseurs\G01_IMDb_DistilBERT_Optimiseurs"

# 3. Créer l'environnement et installer toutes les dépendances
poetry install

# 4. Activer l'environnement Poetry
poetry shell
```

> ⚠️ **Avec Anaconda** : ouvrir d'abord **Anaconda Prompt**, puis exécuter les commandes ci-dessus.  
> Poetry créera son propre environnement virtuel indépendant de conda.

---

### Option B — Avec pip (fallback)

Si Poetry pose problème sur votre machine :

```bash
pip install -r requirements.txt
```

---

## Ordre d'exécution

### Avec Poetry

```bash
# Étape 1 — Pipeline d'entraînement (2-4h sur CPU)
poetry run python main.py

# Étape 2 — Dashboard interactif
poetry run python dashboard.py

# (Optionnel) Notebooks
poetry run jupyter notebook notebooks/exploration.ipynb
poetry run jupyter notebook notebooks/analysis.ipynb
```

### Avec pip (environnement activé)

```bash
# Étape 1 — Pipeline d'entraînement
python main.py

# Étape 2 — Dashboard interactif
python dashboard.py
```

Puis ouvrir dans le navigateur : **http://127.0.0.1:8050/**

> 💡 Le dashboard fonctionne aussi **avant** `main.py` grâce au mode démonstration.

---

## Dashboard — Fonctionnalités

| Onglet | Contenu |
|--------|---------|
| 📈 **Convergence** | Courbes train/val loss & accuracy par optimiseur (sélection interactive) |
| 🏆 **Comparaison** | Barres groupées Val Accuracy / Test F1 pour toutes les configurations |
| 🔍 **Random Search** | Scatter LR vs Accuracy + Heatmap Optimiseur × LR |
| 🌄 **Loss Landscape** | Sharpness par configuration + tableau de platitude |
| ⏱ **Temps / Perf** | Compromis temps d'entraînement vs performance |
| 📋 **Données brutes** | Tableau complet des 9 configurations |

---

## Sorties générées par main.py

| Fichier | Description |
|---------|-------------|
| `outputs/results/random_search_results.csv` | Résultats de toutes les configurations |
| `outputs/results/best_config.json` | Meilleure configuration trouvée |
| `outputs/results/landscape_metrics.csv` | Sharpness par optimiseur |
| `outputs/figures/01_convergence_curves.png` | Courbes train/val loss & accuracy |
| `outputs/figures/02_final_comparison.png` | Comparaison Val Acc / Test F1 |
| `outputs/figures/03_loss_landscape.png` | Profils 1D du loss landscape |
| `outputs/figures/04_heatmap_random_search.png` | Heatmap Optimiseur × LR |
| `outputs/figures/05_sharpness.png` | Sharpness par configuration |
| `outputs/models/<opt>_best.pt` | State dict du meilleur modèle par optimiseur |

---

## Hyperparamètres fixes

| Hyperparamètre | Valeur | Justification |
|----------------|--------|---------------|
| batch_size | 16 | Adapté CPU (< 8 Go RAM) |
| weight_decay | 1e-4 | Régularisation standard |
| max_seq_length | 256 | Compromis couverture / mémoire CPU |
| num_epochs | 3 | Early convergence attendue |
| grad_clip | 1.0 | Stabilité de l'entraînement |

## Espace de recherche

| Hyperparamètre | Distribution | Plage |
|----------------|-------------|-------|
| learning_rate | Log-uniforme | [1e-6, 5e-4] |
| optimizer | Catégorielle | {AdamW, SGD+momentum, Adafactor} |

---

## Formule de Sharpness

```
Sharpness = (1/N) × Σᵢ |L(θ + ε·dᵢ) − L(θ)|
```

Un minima plat (Sharpness ≈ 0) est associé à une meilleure généralisation
(Keskar et al., 2017 ; Foret et al., 2021 — SAM optimizer).

---

## Adaptation CPU

| Contrainte | Solution appliquée |
|------------|-------------------|
| Pas de GPU | torch.float32, torch.set_num_threads(4) |
| RAM limitée | Sous-échantillonnage (1000 train / 200 val par classe) |
| Temps contraint | max_seq_length=256, best_state sauvegardé |
| Loss landscape lourd | n_points=10, n_samples=100, n_dirs=5 |

---

## Références

- Sanh et al. (2019). **DistilBERT, a distilled version of BERT**. arXiv:1910.01108  
- Devlin et al. (2018). **BERT: Pre-training of Deep Bidirectional Transformers**. arXiv:1810.04805  
- Keskar et al. (2017). **On Large-Batch Training for Deep Learning: Generalization Gap**. ICLR 2017  
- Li et al. (2018). **Visualizing the Loss Landscape of Neural Nets**. NeurIPS 2018  
- Loshchilov & Hutter (2019). **Decoupled Weight Decay Regularization (AdamW)**. ICLR 2019  
- Bergstra & Bengio (2012). **Random Search for Hyper-Parameter Optimization**. JMLR  
- HuggingFace Transformers : https://huggingface.co/docs/transformers  
