"""
dashboard.py
------------
Dashboard interactif Dash/Plotly pour le projet G01.
Visualise tous les résultats du Random Search et du Loss Landscape.

Lancement :
    python dashboard.py

Accessible sur : http://127.0.0.1:8050/
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ── Chemins ──────────────────────────────────────────────────────────────────
RESULTS_DIR  = "outputs/results"
FIGURES_DIR  = "outputs/figures"
CSV_PATH     = os.path.join(RESULTS_DIR, "random_search_results.csv")
BEST_PATH    = os.path.join(RESULTS_DIR, "best_config.json")
LANDSCAPE_PATH = os.path.join(RESULTS_DIR, "landscape_metrics.csv")

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "AdamW"    : "#2E75B6",
    "SGD"      : "#E07B39",
    "Adafactor": "#43A047",
    "bg"       : "#F8FAFC",
    "card"     : "#FFFFFF",
    "dark"     : "#1F4E79",
    "text"     : "#2C3E50",
    "border"   : "#DEE2E6",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    """Charge les fichiers de résultats. Retourne des DataFrames vides si absents."""
    results_df  = pd.DataFrame()
    best_config = {}
    landscape_df = pd.DataFrame()

    if os.path.exists(CSV_PATH):
        results_df = pd.read_csv(CSV_PATH)
        # Reconstruire les listes d'historique (stockées comme string)
        for col in ["train_loss_history","val_loss_history",
                    "train_acc_history","val_acc_history"]:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x)

    if os.path.exists(BEST_PATH):
        with open(BEST_PATH) as f:
            best_config = json.load(f)

    if os.path.exists(LANDSCAPE_PATH):
        landscape_df = pd.read_csv(LANDSCAPE_PATH)

    return results_df, best_config, landscape_df


# ─────────────────────────────────────────────────────────────────────────────
#  DONNÉES DE DÉMONSTRATION (si pipeline pas encore exécuté)
# ─────────────────────────────────────────────────────────────────────────────
def make_demo_data():
    """Génère des données de démonstration réalistes pour prévisualisation."""
    np.random.seed(42)
    records = []
    configs = [
        ("AdamW",     2e-5,  [0.68,0.78,0.83], [0.67,0.77,0.81], [0.65,0.75,0.80], [0.66,0.76,0.80]),
        ("AdamW",     5e-5,  [0.71,0.82,0.87], [0.70,0.81,0.86], [0.68,0.79,0.85], [0.69,0.80,0.85]),
        ("AdamW",     1e-4,  [0.74,0.84,0.88], [0.73,0.83,0.87], [0.70,0.81,0.86], [0.71,0.82,0.86]),
        ("SGD",       5e-4,  [0.55,0.62,0.68], [0.54,0.61,0.67], [0.52,0.59,0.65], [0.53,0.60,0.65]),
        ("SGD",       1e-3,  [0.61,0.70,0.76], [0.60,0.69,0.75], [0.58,0.67,0.73], [0.59,0.68,0.73]),
        ("SGD",       5e-3,  [0.58,0.66,0.71], [0.57,0.65,0.70], [0.55,0.63,0.68], [0.56,0.64,0.68]),
        ("Adafactor", 1e-5,  [0.63,0.72,0.78], [0.62,0.71,0.77], [0.60,0.69,0.76], [0.61,0.70,0.76]),
        ("Adafactor", 3e-5,  [0.69,0.79,0.84], [0.68,0.78,0.83], [0.66,0.76,0.82], [0.67,0.77,0.82]),
        ("Adafactor", 1e-4,  [0.72,0.81,0.85], [0.71,0.80,0.84], [0.69,0.78,0.83], [0.70,0.79,0.83]),
    ]
    for i, (opt, lr, tr_acc, va_acc, tr_loss_raw, va_loss_raw) in enumerate(configs):
        tr_loss = [round(1.2-v*0.8+np.random.uniform(-0.02,0.02), 4) for v in tr_acc]
        va_loss = [round(1.3-v*0.8+np.random.uniform(-0.02,0.02), 4) for v in va_acc]
        records.append({
            "optimizer": opt, "learning_rate": lr, "run_idx": (i%3)+1,
            "best_val_acc": max(va_acc), "best_val_f1": max(va_acc)-0.01,
            "final_val_loss": va_loss[-1],
            "test_accuracy": max(va_acc)-0.02, "test_f1": max(va_acc)-0.03,
            "total_time_s": np.random.randint(800,2400),
            "train_loss_history": tr_loss, "val_loss_history": va_loss,
            "train_acc_history": tr_acc,  "val_acc_history": va_acc,
        })
    df = pd.DataFrame(records)
    best = df.loc[df["best_val_acc"].idxmax()]
    best_config = {
        "optimizer": best["optimizer"],
        "learning_rate": float(best["learning_rate"]),
        "best_val_acc":  float(best["best_val_acc"]),
        "test_accuracy": float(best["test_accuracy"]),
        "test_f1":       float(best["test_f1"]),
    }
    landscape = pd.DataFrame([
        {"run_label": "AdamW | LR=1e-04",     "sharpness": 0.0142},
        {"run_label": "SGD | LR=1e-03",       "sharpness": 0.0089},
        {"run_label": "Adafactor | LR=1e-04", "sharpness": 0.0117},
    ])
    return df, best_config, landscape


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPHIQUES PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def fig_convergence(df, selected_opt):
    """Courbes train/val loss et accuracy pour un optimiseur donné."""
    subset = df[df["optimizer"] == selected_opt]
    if subset.empty:
        return go.Figure()

    best_row = subset.loc[subset["best_val_acc"].idxmax()]
    epochs   = list(range(1, len(best_row["train_loss_history"]) + 1))
    color    = COLORS.get(selected_opt, "#888")

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Loss (Train vs Validation)", "Accuracy (Train vs Validation)"))

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=best_row["train_loss_history"],
        mode="lines+markers", name="Train Loss",
        line=dict(color=color, dash="dash", width=2),
        marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=best_row["val_loss_history"],
        mode="lines+markers", name="Val Loss",
        line=dict(color=color, width=2.5),
        marker=dict(size=8, symbol="square")), row=1, col=1)

    # Accuracy
    fig.add_trace(go.Scatter(x=epochs, y=best_row["train_acc_history"],
        mode="lines+markers", name="Train Acc",
        line=dict(color=color, dash="dash", width=2),
        marker=dict(size=8), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=best_row["val_acc_history"],
        mode="lines+markers", name="Val Acc",
        line=dict(color=color, width=2.5),
        marker=dict(size=8, symbol="square"), showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="Époque", tickmode="linear")
    fig.update_yaxes(title_text="Loss",     row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)
    fig.update_layout(
        title=dict(
            text=f"<b>{selected_opt}</b> — Meilleur run (LR = {best_row['learning_rate']:.2e})",
            font=dict(size=15, color=COLORS["dark"])
        ),
        legend=dict(orientation="h", y=-0.15),
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        height=380, margin=dict(t=60, b=60)
    )
    return fig


def fig_comparison(df):
    """Barres groupées : Val Accuracy et Test F1 par configuration."""
    df2 = df.copy()
    df2["config"] = df2["optimizer"] + "<br>" + df2["learning_rate"].apply(lambda x: f"{x:.1e}")
    df2 = df2.sort_values("best_val_acc", ascending=False)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Val Accuracy (meilleure époque)", "Test F1-Macro"))

    for i, (metric, label) in enumerate(
        [("best_val_acc","Val Accuracy"), ("test_f1","Test F1-Macro")], start=1
    ):
        for opt in ["AdamW", "SGD", "Adafactor"]:
            sub = df2[df2["optimizer"] == opt]
            fig.add_trace(go.Bar(
                x=sub["config"], y=sub[metric],
                name=opt, marker_color=COLORS.get(opt,"#888"),
                showlegend=(i==1),
                text=[f"{v:.3f}" for v in sub[metric]],
                textposition="outside"
            ), row=1, col=i)

    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Comparaison finale</b> — Toutes les configurations",
                   font=dict(size=15, color=COLORS["dark"])),
        legend=dict(orientation="h", y=-0.2),
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        height=420, margin=dict(t=60, b=80)
    )
    return fig


def fig_heatmap(df):
    """Heatmap Val Accuracy : optimiseur × learning rate."""
    df2 = df.copy()
    df2["lr_label"] = df2["learning_rate"].apply(lambda x: f"{x:.2e}")
    pivot = df2.pivot_table(index="optimizer", columns="lr_label",
                            values="best_val_acc", aggfunc="max")
    fig = go.Figure(go.Heatmap(
        z=pivot.values.tolist(),
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="YlGnBu",
        text=[[f"{v:.3f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorbar=dict(title="Val Acc"),
    ))
    fig.update_layout(
        title=dict(text="<b>Heatmap Random Search</b> — Val Accuracy par Optimiseur × LR",
                   font=dict(size=15, color=COLORS["dark"])),
        xaxis_title="Learning Rate",
        yaxis_title="Optimiseur",
        paper_bgcolor="white",
        height=320, margin=dict(t=60, b=60)
    )
    return fig


def fig_sharpness(landscape_df):
    """Barres horizontales de la Sharpness."""
    if landscape_df.empty:
        return go.Figure()

    df2 = landscape_df.sort_values("sharpness", ascending=True)
    # Couleur selon l'optimiseur
    bar_colors = []
    for label in df2["run_label"]:
        matched = "#888"
        for opt, col in COLORS.items():
            if opt in label:
                matched = col
                break
        bar_colors.append(matched)

    fig = go.Figure(go.Bar(
        x=df2["sharpness"], y=df2["run_label"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.5f}" for v in df2["sharpness"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="<b>Sharpness du Loss Landscape</b> — (↓ = minimum plus plat = meilleure généralisation)",
                   font=dict(size=14, color=COLORS["dark"])),
        xaxis_title="Sharpness = (1/N) Σ |L(θ + ε·dᵢ) − L(θ)|",
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        height=300, margin=dict(t=60, b=60, l=220)
    )
    return fig


def fig_scatter_lr(df):
    """Scatter : LR (log) vs Val Accuracy coloré par optimiseur."""
    fig = go.Figure()
    for opt in ["AdamW", "SGD", "Adafactor"]:
        sub = df[df["optimizer"] == opt]
        fig.add_trace(go.Scatter(
            x=sub["learning_rate"], y=sub["best_val_acc"],
            mode="markers+text",
            name=opt,
            marker=dict(size=16, color=COLORS.get(opt,"#888"),
                        line=dict(color="white", width=2)),
            text=[f"{v:.3f}" for v in sub["best_val_acc"]],
            textposition="top center",
        ))
    fig.update_xaxes(type="log", title_text="Learning Rate (échelle logarithmique)")
    fig.update_yaxes(title_text="Val Accuracy", range=[0, 1.05])
    fig.update_layout(
        title=dict(text="<b>Random Search</b> — Val Accuracy en fonction du Learning Rate",
                   font=dict(size=15, color=COLORS["dark"])),
        legend=dict(orientation="h", y=-0.15),
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        height=380, margin=dict(t=60, b=60)
    )
    return fig


def fig_time_perf(df):
    """Scatter : Temps entraînement vs Performance."""
    fig = go.Figure()
    for opt in ["AdamW", "SGD", "Adafactor"]:
        sub = df[df["optimizer"] == opt]
        fig.add_trace(go.Scatter(
            x=sub["total_time_s"]/60, y=sub["test_accuracy"],
            mode="markers",
            name=opt,
            marker=dict(size=14, color=COLORS.get(opt,"#888"),
                        line=dict(color="white", width=2)),
            hovertemplate=(
                f"<b>{opt}</b><br>"
                "LR: %{customdata:.2e}<br>"
                "Temps: %{x:.1f} min<br>"
                "Test Acc: %{y:.3f}"
            ),
            customdata=sub["learning_rate"],
        ))
    fig.update_layout(
        title=dict(text="<b>Compromis Temps / Performance</b>",
                   font=dict(size=15, color=COLORS["dark"])),
        xaxis_title="Temps d'entraînement (minutes)",
        yaxis_title="Test Accuracy",
        legend=dict(orientation="h", y=-0.15),
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        height=350, margin=dict(t=60, b=60)
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CARTES KPI
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(title, value, subtitle="", color=COLORS["dark"]):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1",
                   style={"fontSize":"12px","fontWeight":"600","textTransform":"uppercase"}),
            html.H3(value, style={"color":color,"fontWeight":"800","marginBottom":"2px"}),
            html.P(subtitle, className="text-muted mb-0",
                   style={"fontSize":"11px"}),
        ])
    ], style={
        "border":f"1px solid {COLORS['border']}",
        "borderTop":f"4px solid {color}",
        "borderRadius":"8px","background":"white","padding":"4px"
    })


# ─────────────────────────────────────────────────────────────────────────────
#  APPLICATION DASH
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="G01 — Dashboard NLP"
)

# ── Chargement initial des données ────────────────────────────────────────────
results_df, best_config, landscape_df = load_data()
is_demo = results_df.empty
if is_demo:
    results_df, best_config, landscape_df = make_demo_data()

# ── Calcul des KPIs ───────────────────────────────────────────────────────────
best_val  = results_df["best_val_acc"].max()
best_test = results_df["test_accuracy"].max()
best_f1   = results_df["test_f1"].max()
best_opt  = best_config.get("optimizer", "—")
best_lr   = best_config.get("learning_rate", 0)
n_configs = len(results_df)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, style={"background":COLORS["bg"],"minHeight":"100vh"}, children=[

    # ── BANDEAU HEADER ──────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        html.Div([
            html.H2("G01 — Fine-tuning de Transformers",
                    style={"color":"white","fontWeight":"800","marginBottom":"4px"}),
            html.P("DistilBERT · IMDb · P01 Benchmark d'Optimiseurs · Random Search",
                   style={"color":"rgba(255,255,255,0.85)","marginBottom":"0","fontSize":"14px"}),
        ], style={"flex":"1"}),
        html.Div([
            html.Span("DEMO DATA" if is_demo else "LIVE DATA",
                style={
                    "background":"#FFC107" if is_demo else "#28A745",
                    "color":"white","borderRadius":"20px","padding":"4px 14px",
                    "fontWeight":"700","fontSize":"12px","marginRight":"12px"
                }),
            html.Span("Cours : Optimisation & Loss Landscape",
                style={"color":"rgba(255,255,255,0.7)","fontSize":"12px"}),
        ]),
    ], style={"display":"flex","alignItems":"center","justifyContent":"space-between",
              "padding":"20px 30px",
              "background":f"linear-gradient(135deg, {COLORS['dark']}, {COLORS['AdamW']})"}))
    , className="mb-0"),

    # ── ALERTE DEMO ─────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(
        dbc.Alert([
            html.I(className="me-2"),
            html.Strong("Mode démonstration : "),
            "Les données affichées sont simulées. Exécutez ",
            html.Code("python main.py"),
            " puis relancez ",
            html.Code("python dashboard.py"),
            " pour voir vos vrais résultats."
        ], color="warning", dismissable=True,
           style={"margin":"12px 0","fontSize":"13px"})
    ), style={"padding":"0 16px"}) if is_demo else html.Div(),

    # ── KPI CARDS ────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(kpi_card("Meilleur Optimiseur", best_opt,
                         f"LR = {best_lr:.2e}", COLORS.get(best_opt, COLORS["dark"])), md=2),
        dbc.Col(kpi_card("Val Accuracy Max", f"{best_val:.4f}",
                         "Meilleure config", COLORS["AdamW"]), md=2),
        dbc.Col(kpi_card("Test Accuracy", f"{best_test:.4f}",
                         "Set de test", COLORS["Adafactor"]), md=2),
        dbc.Col(kpi_card("Test F1-Macro", f"{best_f1:.4f}",
                         "Macro-average", COLORS["SGD"]), md=2),
        dbc.Col(kpi_card("Configurations", str(n_configs),
                         "3 optimiseurs × 3 LR", "#6C757D"), md=2),
        dbc.Col(kpi_card("Dataset", "IMDb D01",
                         "50k critiques · 2 classes", "#495057"), md=2),
    ], className="g-3", style={"padding":"16px 16px 0"}),

    html.Hr(style={"margin":"16px","borderColor":COLORS["border"]}),

    # ── ONGLETS ───────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(
        dcc.Tabs(id="tabs", value="tab-convergence",
            style={"fontFamily":"Arial","fontWeight":"600"},
            children=[
                dcc.Tab(label="📈 Convergence",     value="tab-convergence"),
                dcc.Tab(label="🏆 Comparaison",     value="tab-comparison"),
                dcc.Tab(label="🔍 Random Search",   value="tab-search"),
                dcc.Tab(label="🌄 Loss Landscape",  value="tab-landscape"),
                dcc.Tab(label="⏱ Temps / Perf",    value="tab-time"),
                dcc.Tab(label="📋 Données brutes",  value="tab-data"),
            ])
    ), style={"padding":"0 16px"}),

    # ── CONTENU ONGLETS ───────────────────────────────────────────────────────
    html.Div(id="tab-content", style={"padding":"16px"}),

])


# ─────────────────────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):

    # ── TAB 1 : Convergence ──────────────────────────────────────────────────
    if tab == "tab-convergence":
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Strong("Sélectionner un optimiseur")),
                    dbc.CardBody(
                        dcc.RadioItems(
                            id="opt-selector",
                            options=[{"label":f" {o}","value":o}
                                     for o in ["AdamW","SGD","Adafactor"]],
                            value="AdamW",
                            inputStyle={"marginRight":"8px"},
                            labelStyle={"display":"block","padding":"6px 0",
                                        "fontWeight":"600","fontSize":"14px"},
                        )
                    )
                ], style={"borderTop":f"4px solid {COLORS['dark']}"}),
                html.Br(),
                dbc.Card([
                    dbc.CardHeader(html.Strong("Meilleur run")),
                    dbc.CardBody(html.Div(id="best-run-info"))
                ])
            ], md=3),
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(id="convergence-graph",
                                                config={"displayModeBar":False})))
            ], md=9)
        ])

    # ── TAB 2 : Comparaison ──────────────────────────────────────────────────
    elif tab == "tab-comparison":
        return dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_comparison(results_df),
                          config={"displayModeBar":False})
            )), md=12),
            dbc.Col([
                html.Br(),
                dbc.Card([
                    dbc.CardHeader(html.Strong("Interprétation")),
                    dbc.CardBody([
                        html.P("Ce graphique compare toutes les configurations testées "
                               "selon leur Val Accuracy et Test F1-Macro.", className="mb-2"),
                        html.P([html.Strong("AdamW "), "est généralement l'optimiseur de référence "
                               "pour le fine-tuning de Transformers."], className="mb-2"),
                        html.P([html.Strong("SGD "), "peut nécessiter un learning rate plus élevé "
                               "que les méthodes adaptatives."], className="mb-1"),
                        html.P([html.Strong("Adafactor "), "offre une bonne alternative à AdamW "
                               "avec une empreinte mémoire réduite."], className="mb-0"),
                    ])
                ])
            ], md=12)
        ])

    # ── TAB 3 : Random Search ────────────────────────────────────────────────
    elif tab == "tab-search":
        return dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_scatter_lr(results_df),
                          config={"displayModeBar":False})
            )), md=7),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_heatmap(results_df),
                          config={"displayModeBar":False})
            )), md=5),
            dbc.Col([
                html.Br(),
                dbc.Card([
                    dbc.CardHeader(html.Strong("Paramètres du Random Search")),
                    dbc.CardBody([
                        html.Ul([
                            html.Li("Distribution : Log-uniforme dans [1e-6, 5e-4]"),
                            html.Li("3 tirages par optimiseur (graine = 42)"),
                            html.Li("Total : 9 configurations évaluées"),
                            html.Li("Critère de sélection : Val Accuracy maximale"),
                        ], style={"fontSize":"13px"})
                    ])
                ])
            ], md=12)
        ])

    # ── TAB 4 : Loss Landscape ───────────────────────────────────────────────
    elif tab == "tab-landscape":
        return dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_sharpness(landscape_df),
                          config={"displayModeBar":False})
            )), md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Strong("Formule — Sharpness")),
                    dbc.CardBody([
                        html.P("Sharpness = (1/N) × Σᵢ |L(θ + ε·dᵢ) − L(θ)|",
                               style={"fontFamily":"Courier New","fontSize":"13px",
                                      "background":"#F0F0F0","padding":"10px",
                                      "borderRadius":"4px","fontWeight":"bold"}),
                        html.Hr(),
                        html.P([html.Strong("Un minimum plat "), "(sharpness ≈ 0) est associé à une ",
                               html.Strong("meilleure généralisation")], className="mb-2"),
                        html.P("ε = 0.05 | N = 5 directions | 100 exemples / évaluation",
                               style={"fontSize":"12px","color":"#666"}),
                        html.Hr(),
                        html.P([html.Strong("Référence : "),
                               "Keskar et al. (2017) — On Large-Batch Training"],
                               style={"fontSize":"12px","fontStyle":"italic"}),
                    ])
                ])
            ], md=4),

            # Tableau sharpness
            dbc.Col([
                html.Br(),
                dbc.Card([
                    dbc.CardHeader(html.Strong("Détail des métriques de platitude")),
                    dbc.CardBody(
                        html.Table([
                            html.Thead(html.Tr([
                                html.Th("Configuration", style={"padding":"8px"}),
                                html.Th("Sharpness",     style={"padding":"8px"}),
                                html.Th("Platitude",     style={"padding":"8px"}),
                            ], style={"background":COLORS["dark"],"color":"white"})),
                            html.Tbody([
                                html.Tr([
                                    html.Td(row["run_label"], style={"padding":"8px"}),
                                    html.Td(f"{row['sharpness']:.5f}", style={"padding":"8px"}),
                                    html.Td(
                                        "🟢 Plat" if row["sharpness"] < 0.01
                                        else ("🟡 Moyen" if row["sharpness"] < 0.015
                                              else "🔴 Pointu"),
                                        style={"padding":"8px"}
                                    ),
                                ], style={"background": "#F8FAFC" if i%2==0 else "white"})
                                for i, row in landscape_df.iterrows()
                            ] if not landscape_df.empty else [
                                html.Tr([html.Td("Données non disponibles", colSpan=3,
                                               style={"padding":"12px","textAlign":"center","color":"#666"})])
                            ])
                        ], style={"width":"100%","borderCollapse":"collapse","fontSize":"13px"})
                    )
                ])
            ], md=12)
        ])

    # ── TAB 5 : Temps / Performance ──────────────────────────────────────────
    elif tab == "tab-time":
        return dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_time_perf(results_df),
                          config={"displayModeBar":False})
            )), md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Strong("Temps moyen par optimiseur")),
                    dbc.CardBody(
                        html.Div([
                            html.Div([
                                html.Div(style={"width":"14px","height":"14px","borderRadius":"50%",
                                               "background":COLORS.get(opt,"#888"),"marginRight":"8px"}),
                                html.Strong(f"{opt} : ", style={"marginRight":"4px"}),
                                html.Span(
                                    f"{results_df[results_df['optimizer']==opt]['total_time_s'].mean()/60:.0f} min"
                                    if not results_df[results_df['optimizer']==opt].empty else "—"
                                ),
                            ], style={"display":"flex","alignItems":"center","padding":"8px 0",
                                     "borderBottom":"1px solid #eee"})
                            for opt in ["AdamW","SGD","Adafactor"]
                        ])
                    )
                ]),
                html.Br(),
                dbc.Card([
                    dbc.CardHeader(html.Strong("Note CPU")),
                    dbc.CardBody([
                        html.P("Les temps sont estimés sur CPU (sans GPU).",
                               className="mb-1", style={"fontSize":"13px"}),
                        html.P("Optimisations appliquées :", className="mb-1",
                               style={"fontSize":"13px","fontWeight":"600"}),
                        html.Ul([
                            html.Li("torch.set_num_threads(4)"),
                            html.Li("batch_size = 16"),
                            html.Li("max_seq_length = 256"),
                            html.Li("sous-échantillonnage 2000 ex."),
                        ], style={"fontSize":"12px","color":"#555"})
                    ])
                ])
            ], md=4)
        ])

    # ── TAB 6 : Données brutes ────────────────────────────────────────────────
    elif tab == "tab-data":
        cols = ["optimizer","learning_rate","run_idx","best_val_acc",
                "test_accuracy","test_f1","total_time_s"]
        display_df = results_df[[c for c in cols if c in results_df.columns]].copy()
        display_df["learning_rate"] = display_df["learning_rate"].apply(lambda x: f"{x:.2e}")
        display_df = display_df.sort_values("best_val_acc", ascending=False)
        display_df.columns = ["Optimiseur","Learning Rate","Run","Val Acc","Test Acc","Test F1","Temps (s)"]

        rows = []
        for i, row in display_df.iterrows():
            is_best = row["Val Acc"] == display_df["Val Acc"].max()
            rows.append(html.Tr(
                [html.Td(str(v), style={"padding":"10px 12px",
                    "fontWeight":"700" if is_best else "normal",
                    "color":COLORS["dark"] if is_best else COLORS["text"]})
                 for v in row.values],
                style={"background":"#EBF5FB" if is_best else ("#F8FAFC" if i%2==0 else "white")}
            ))

        return dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.Strong("Tableau complet des résultats "),
                    html.Span("(ligne surlignée = meilleure configuration)",
                             style={"fontSize":"12px","color":"#666","fontWeight":"normal"})
                ]),
                dbc.CardBody(
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th(c, style={"padding":"10px 12px","background":COLORS["dark"],
                                             "color":"white","fontWeight":"700"})
                            for c in display_df.columns
                        ])),
                        html.Tbody(rows)
                    ], style={"width":"100%","borderCollapse":"collapse",
                              "fontSize":"13px","fontFamily":"Arial"})
                )
            ]), md=12)
        ])

    return html.Div("Onglet non trouvé")


# ── Callback : Convergence ────────────────────────────────────────────────────
@app.callback(
    Output("convergence-graph","figure"),
    Output("best-run-info","children"),
    Input("opt-selector","value")
)
def update_convergence(selected_opt):
    fig = fig_convergence(results_df, selected_opt)

    subset = results_df[results_df["optimizer"] == selected_opt]
    if subset.empty:
        return fig, html.P("Aucune donnée")

    best = subset.loc[subset["best_val_acc"].idxmax()]
    info = [
        html.P([html.Strong("LR : "), f"{best['learning_rate']:.2e}"],
               style={"marginBottom":"4px","fontSize":"13px"}),
        html.P([html.Strong("Val Acc : "), f"{best['best_val_acc']:.4f}"],
               style={"marginBottom":"4px","fontSize":"13px"}),
        html.P([html.Strong("Test Acc : "), f"{best['test_accuracy']:.4f}"],
               style={"marginBottom":"4px","fontSize":"13px"}),
        html.P([html.Strong("Test F1 : "), f"{best['test_f1']:.4f}"],
               style={"marginBottom":"4px","fontSize":"13px"}),
        html.P([html.Strong("Temps : "), f"{best['total_time_s']/60:.0f} min"],
               style={"marginBottom":"0","fontSize":"13px"}),
    ]
    return fig, info


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  G01 — Dashboard Résultats")
    print("  Accessible sur : http://127.0.0.1:8050/")
    print("="*55 + "\n")
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
