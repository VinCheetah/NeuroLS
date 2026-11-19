# ruff: noqa: E402
"""Interactive Streamlit Dashboard for Neural Local Search Model Explainability.

Enables:
- Dynamic discovery and fine-grained manual control of all model input features.
- Single-action scoring, linear weight decomposition, and autograd gradient sensitivities.
- 1D response curves S(x) and 2D interaction surfaces for per-action networks.
- Real-world validation on actual NK Landscape coordinates with interactive stepping
  (following the model's choice or manual bit flips), single or multi-agent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure neurols is on the Python path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import plotly.graph_objects as go
import streamlit as st

from neurols.analysis.interpretability import (
    ModelInspector,
    RealNKSession,
    sample_feature_value,
)
from neurols.strategies.neural.neural_strategy import NeuralLocalSearchStrategy
from neurols.training.registry import list_saved_models

# ============================================================================
# Streamlit Page Setup & Custom Styling
# ============================================================================

st.set_page_config(
    page_title="NeuroLS • Model Reason & Interpretability",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    .score-badge-pos {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .score-badge-neg {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .bit-badge-1 {
        background-color: #3B82F6;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-family: monospace;
        margin-right: 2px;
        display: inline-block;
    }
    .bit-badge-0 {
        background-color: #E2E8F0;
        color: #475569;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-family: monospace;
        margin-right: 2px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Session State Management
# ============================================================================


def init_session_state() -> None:
    """Initialize state dictionaries for features, active models, and real NK session."""
    if "features_dict" not in st.session_state:
        st.session_state.features_dict = {}
    if "current_model_key" not in st.session_state:
        st.session_state.current_model_key = None
    if "nk_session" not in st.session_state:
        st.session_state.nk_session = None
    if "selected_agent_idx" not in st.session_state:
        st.session_state.selected_agent_idx = 0


init_session_state()


# ============================================================================
# Model Loading & Inspector Resolution
# ============================================================================


@st.cache_resource
def load_inspector(model_path_or_name: str, n_agents: int = 1) -> ModelInspector:
    """Load or instantiate ModelInspector with caching."""
    if model_path_or_name.startswith("CUSTOM:"):
        # Custom on-the-fly model format: CUSTOM:policy_type:features_spec:n_actions
        parts = model_path_or_name.split(":")
        p_type = parts[1]
        feats = parts[2]
        n_act = int(parts[3])
        strat = NeuralLocalSearchStrategy(
            policy_type=p_type,  # type: ignore
            features=feats,
            n_actions=n_act,
            n_agents=n_agents,
        )
        return ModelInspector(strat, n_agents=n_agents)
    else:
        # Load from local registry checkpoint
        return ModelInspector.from_checkpoint(
            model_path_or_name,
            override_n_agents=n_agents if n_agents > 1 else None,
        )


# ============================================================================
# Sidebar: Model, Agents & NK Configuration
# ============================================================================

st.sidebar.markdown("## 🧠 Sélection du Modèle")

# Discover all saved checkpoints
saved_models = list_saved_models("checkpoints")
model_options: dict[str, str] = {}
for m in saved_models:
    display = f"📦 {m.experiment_name}/{m.model_name} ({m.policy_type}, M={m.n_agents}, N={m.n})"
    ref = str(m.file_path) if m.file_path else f"{m.experiment_name}:{m.model_name}"
    model_options[display] = ref

model_options["🛠️ Créer un Modèle Personnalisé (À la volée)"] = "CUSTOM"

selected_model_label = st.sidebar.selectbox(
    "Modèle à Inspecter :",
    options=list(model_options.keys()),
    index=0 if saved_models else len(model_options) - 1,
)

model_ref = model_options[selected_model_label]

# Custom model creation UI if selected
if model_ref == "CUSTOM":
    st.sidebar.markdown("### Configuration Personnalisée")
    custom_policy = st.sidebar.selectbox(
        "Type d'Architecture :",
        ["linear", "mlp", "per_action", "deepsets", "invariant_nnet", "peer_invariant_per_action"],
        index=0,
    )
    custom_features = st.sidebar.text_input(
        "Spécification des Features :",
        value="@action_basic,best_locus,dist_others_barycentre,normalized_rank",
        help="Ex: 'delta,ranked_delta', '@action_basic', '@all'",
    )
    custom_n = st.sidebar.number_input("Nombre d'actions (N) :", min_value=4, max_value=128, value=12)
    model_ref = f"CUSTOM:{custom_policy}:{custom_features}:{custom_n}"

# Multi-Agent Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("## 👥 Multi-Agents & Problème NK")
n_agents = st.sidebar.slider(
    "Nombre d'Agents (M) :",
    min_value=1,
    max_value=8,
    value=1,
    help="Si M > 1, active les métriques multi-agents (distance aux pairs, barycentres, consensus, classements).",
)

# NK Landscape parameters
nk_n = st.sidebar.number_input(
    "Dimension N (bits) :",
    min_value=4,
    max_value=64,
    value=12,
    help="Nombre de variables binaires dans le problème NK.",
)
nk_k = st.sidebar.slider(
    "Degré d'Epistasie K :",
    min_value=0,
    max_value=max(nk_n - 1, 0),
    value=min(2, max(nk_n - 1, 0)),
    help="Nombre d'interactions épistatiques par locus (K=0: linéaire, K élevé: très accidenté).",
)
nk_topology = st.sidebar.selectbox(
    "Topologie des Interactions :",
    ["adjacent", "random"],
    index=0,
)
nk_seed = st.sidebar.number_input("Graine Aléatoire (Seed NK) :", value=42, step=1)

if st.sidebar.button("🎲 Nouveau Paysage NK & Redémarrage"):
    st.session_state.nk_session = None

# Load Inspector
try:
    inspector = load_inspector(model_ref, n_agents=n_agents)
except Exception as e:
    st.sidebar.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

# Detect model change to reset features
if st.session_state.current_model_key != model_ref:
    st.session_state.current_model_key = model_ref
    st.session_state.features_dict = inspector.get_default_feature_vector()
    st.session_state.nk_session = None

# Ensure all current feature names exist in session state
default_features = inspector.get_default_feature_vector()
for feat_name, default_val in default_features.items():
    if feat_name not in st.session_state.features_dict:
        st.session_state.features_dict[feat_name] = default_val

# Sidebar Model Specs Overview
with st.sidebar.expander("ℹ️ Spécifications du Réseau", expanded=False):
    st.markdown(f"**Politique :** `{inspector.strategy.policy_type}`")
    st.markdown(f"**Mode d'Évaluation :** `{inspector.strategy.eval_mode.value}`")
    st.markdown(f"**Nombre de Paramètres :** `{inspector.strategy.policy.num_parameters}`")
    st.markdown(f"**Nombre d'Entrées (F) :** `{len(inspector.feature_names)}`")
    if inspector.manifest:
        st.markdown(f"**Meilleure Fitness Entraînée :** `{inspector.manifest.best_fitness:.4f}`")
        st.markdown(f"**Générations :** `{inspector.manifest.metrics.get('total_generations', 'N/A')}`")


# ============================================================================
# Main Header
# ============================================================================

st.markdown('<div class="main-title">🧠 NeuroLS • Studio d\'Interprétabilité & Raisonnement</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Inspectez et contrôlez chaque entrée du réseau, analysez les courbes de sensibilité '
    'et testez la politique en conditions réelles sur un paysage NK interactif.</div>',
    unsafe_allow_html=True,
)


# ============================================================================
# Main Tabs
# ============================================================================

tab_control, tab_curves, tab_real_nk = st.tabs([
    "🎛️ Contrôle des Entrées & Décision",
    "📈 Analyse de Sensibilité & Courbe Per-Action",
    "🌐 Test en Conditions Réelles (NK)",
])


# ============================================================================
# TAB 1: Contrôle des Entrées & Décision du Réseau
# ============================================================================

with tab_control:
    st.markdown("### 🎛️ Contrôle Séparé de Chaque Entrée du Réseau")
    st.caption(
        "Ce panneau s'adapte automatiquement aux features du modèle. Réglez chaque feature indépendamment, "
        "testez des combinaisons spécifiques ou lancez des perturbations aléatoires pour comprendre la logique du réseau."
    )

    # Global Randomization / Reset Toolbar
    tb_c1, tb_c2, tb_c3, tb_c4 = st.columns([1.5, 1.5, 1.5, 3.5])
    with tb_c1:
        if st.button("🎲 Tout Aléatoire", use_container_width=True):
            st.session_state.features_dict = inspector.sample_random_feature_vector()
            st.rerun()
    with tb_c2:
        if st.button("↺ Remise à Zéro", use_container_width=True):
            for k in st.session_state.features_dict:
                st.session_state.features_dict[k] = 0.0
            st.rerun()
    with tb_c3:
        if st.button("🎯 Valeurs Typiques", use_container_width=True):
            st.session_state.features_dict = inspector.get_default_feature_vector()
            st.rerun()

    st.markdown("---")

    col_inputs, col_output = st.columns([1.3, 1.0], gap="large")

    # Left: Dynamic Input Sliders & Individual Randomizers
    with col_inputs:
        st.markdown("#### ⚙️ Entrées du Modèle")

        # Group features by category
        action_feats = [info for info in inspector.feature_infos if info.category == "action"]
        agent_feats = [info for info in inspector.feature_infos if info.category == "agent"]
        context_feats = [info for info in inspector.feature_infos if info.category == "context"]

        def render_feature_group(group_title: str, infos: list[Any]) -> None:
            if not infos:
                return
            st.markdown(f"##### {group_title}")
            for info in infos:
                f_name = info.name
                cur_val = float(st.session_state.features_dict.get(f_name, info.default_val))

                fc1, fc2 = st.columns([4.2, 0.8])
                with fc1:
                    if info.is_binary:
                        val = st.radio(
                            f"{f_name} : {info.description}",
                            options=[0.0, 1.0],
                            index=0 if cur_val == 0.0 else 1,
                            horizontal=True,
                            key=f"slider_{f_name}",
                            help=f"Formule: {info.formula}",
                        )
                    elif info.is_discrete:
                        val = st.radio(
                            f"{f_name} : {info.description}",
                            options=[-1.0, 1.0],
                            index=0 if cur_val <= 0.0 else 1,
                            horizontal=True,
                            key=f"slider_{f_name}",
                            help=f"Formule: {info.formula}",
                        )
                    else:
                        val = st.slider(
                            f"{f_name} : {info.description}",
                            min_value=float(info.min_val),
                            max_value=float(info.max_val),
                            value=float(cur_val),
                            step=float(info.step_val),
                            key=f"slider_{f_name}",
                            help=f"Formule: {info.formula}",
                        )
                    st.session_state.features_dict[f_name] = float(val)

                with fc2:
                    st.write("")  # Spacing alignment
                    st.write("")
                    if st.button("🎲", key=f"rnd_{f_name}", help=f"Tirer une valeur aléatoire pour {f_name}"):
                        st.session_state.features_dict[f_name] = sample_feature_value(f_name)
                        st.rerun()

        render_feature_group("🎯 Features Spécifiques à l'Action", action_feats)
        if agent_feats:
            render_feature_group("👥 Features d'Agent & Multi-Agents", agent_feats)
        if context_feats:
            render_feature_group("⏱️ Features de Contexte Temporel & Stagnation", context_feats)

    # Right: Network Output & Explainability Breakdown
    with col_output:
        st.markdown("#### 📊 Résultat & Raisonnement du Réseau")

        # Evaluate model on current feature values
        eval_result = inspector.evaluate_single_action(st.session_state.features_dict)

        # Big Metric Display
        score_val = eval_result.score
        score_class = "score-badge-pos" if eval_result.is_positive else "score-badge-neg"
        verdict = "FAVORISÉE (CANDIDATE AU CHOIX)" if eval_result.is_positive else "DÉFAVORISÉE / REJETÉE"

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">SCORE UNIQUE (LOGIT DU RÉSEAU)</div>
                <div style="font-size: 2.5rem; font-weight: 800; margin: 0.2rem 0;">{score_val:+.4f}</div>
                <div><span class="{score_class}">Verdict: {verdict}</span></div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748B;">
                    Équivalent Sigmoïde σ(Score) : <b>{eval_result.confidence_probability:.2%}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Linear Breakdown Waterfall if Linear
        if eval_result.linear_breakdown is not None:
            st.markdown("##### 🔍 Décomposition Linéaire des Contributions")
            st.caption("Chaque composante contribue exactement par : Contribution = Poids w_i × Valeur x_i.")

            items = list(eval_result.linear_breakdown.items())
            if eval_result.linear_bias is not None:
                items.append(("Biais (w0)", eval_result.linear_bias))

            names = [it[0] for it in items]
            contribs = [it[1] for it in items]
            colors = ["#10B981" if c >= 0 else "#EF4444" for c in contribs]

            fig_linear = go.Figure(
                go.Bar(
                    x=contribs,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{c:+.3f}" for c in contribs],
                    textposition="auto",
                )
            )
            fig_linear.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                xaxis_title="Contribution au Score (Logit)",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_linear, use_container_width=True)

        # Autograd Sensitivity Saliency Bar Chart
        st.markdown("##### ⚡ Sensibilité Locale du Réseau (Gradient Absolu)")
        st.caption("Mesure de l'impact instantané de chaque entrée : |d(Score) / d(Feature)|.")

        sens_items = sorted(eval_result.feature_sensitivities.items(), key=lambda x: x[1], reverse=True)
        s_names = [it[0] for it in sens_items]
        s_vals = [it[1] for it in sens_items]

        fig_sens = go.Figure(
            go.Bar(
                x=s_vals,
                y=s_names,
                orientation="h",
                marker_color="#6366F1",
                text=[f"{v:.3f}" for v in s_vals],
                textposition="auto",
            )
        )
        fig_sens.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            xaxis_title="Sensibilité |d(Score)/d(Feature)|",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_sens, use_container_width=True)


# ============================================================================
# TAB 2: Analyse de Sensibilité & Courbe Per-Action
# ============================================================================

with tab_curves:
    st.markdown("### 📈 Analyse de Sensibilité & Courbe de Réponse (Réseaux Per-Action)")
    st.caption(
        "Tracez la fonction de score unique S(x) en faisant varier une entrée (ex: delta_fitness, rank, distance aux pairs) "
        "tout en maintenant les autres entrées aux valeurs réglées dans le Tab 1. "
        "Permet d'identifier précisément les seuils de décision (passage à zéro) et la non-linéarité apprise par le modèle."
    )

    sc_c1, sc_c2, sc_c3 = st.columns([2, 1.5, 1.5])

    with sc_c1:
        target_feat = st.selectbox(
            "Entrée à Faire Varier (Axe X) :",
            options=inspector.feature_names,
            index=0,
            help="Sélectionnez l'entrée dont vous souhaitez analyser la courbe de réponse.",
        )

    target_info = inspector.get_feature_info(target_feat)

    with sc_c2:
        sweep_min = st.number_input(
            "Borne Inférieure X_min :",
            value=float(target_info.min_val),
            step=0.1,
        )
    with sc_c3:
        sweep_max = st.number_input(
            "Borne Supérieure X_max :",
            value=float(target_info.max_val),
            step=0.1,
        )

    # Compute 1D sensitivity curve
    curve_res = inspector.compute_sensitivity_curve(
        target_feature=target_feat,
        base_features=st.session_state.features_dict,
        min_val=sweep_min,
        max_val=sweep_max,
        num_points=120,
    )

    # Key Metrics Banner
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.metric("Valeur Courante X", f"{curve_res.current_x:.3f}")
    with m_c2:
        st.metric("Score Courant S(X)", f"{curve_res.current_score:+.4f}")
    with m_c3:
        st.metric(
            "Pente Locale dS/dX",
            f"{curve_res.local_gradient:+.4f}",
            help="Dérivée numérique au point de fonctionnement actuel.",
        )
    with m_c4:
        crossing_str = (
            ", ".join(f"{z:.3f}" for z in curve_res.zero_crossings)
            if curve_res.zero_crossings
            else "Aucun (Score toujours du même signe)"
        )
        st.metric("Seuil(s) de Décision (S=0)", crossing_str)

    # Interactive Plotly Curve
    fig_curve = go.Figure()

    # Zero score reference line
    fig_curve.add_hline(
        y=0,
        line_dash="dash",
        line_color="#94A3B8",
        annotation_text="Seuil de Décision (S=0)",
        annotation_position="bottom right",
    )

    # Main response curve
    fig_curve.add_trace(
        go.Scatter(
            x=curve_res.x_values,
            y=curve_res.scores,
            mode="lines",
            name="Score S(X)",
            line=dict(color="#2563EB", width=3),
            hovertemplate="<b>%{x:.3f}</b> → Score: <b>%{y:+.4f}</b><extra></extra>",
        )
    )

    # Current operating point marker
    fig_curve.add_trace(
        go.Scatter(
            x=[curve_res.current_x],
            y=[curve_res.current_score],
            mode="markers",
            name="Point Actuel (Tab 1)",
            marker=dict(color="#DC2626", size=12, symbol="diamond"),
            hovertemplate="Point Actuel: (%{x:.3f}, %{y:+.4f})<extra></extra>",
        )
    )

    # Zero crossing vertical markers
    for z in curve_res.zero_crossings:
        fig_curve.add_vline(
            x=z,
            line_dash="dot",
            line_color="#10B981",
            annotation_text=f"Bascule x={z:.3f}",
            annotation_position="top left",
        )

    fig_curve.update_layout(
        title=f"Réponse de la Politique en Fonction de '{target_feat}'",
        xaxis_title=f"Valeur de l'Entrée : {target_feat}",
        yaxis_title="Score Unique de l'Action (Logit)",
        template="plotly_white",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=40, b=40),
        height=450,
    )

    st.plotly_chart(fig_curve, use_container_width=True)

    # Frozen features note
    with st.expander("📌 Valeurs Fixes des Autres Entrées (Condition Actuelle)"):
        st.write(curve_res.frozen_features)

    # Bonus: 2D Interaction Surface / Heatmap
    with st.expander("🌐 Explorer une Surface d'Interaction 2D (Synergie Entre Deux Features)"):
        st.markdown(
            "Analysez comment le modèle combine deux features ensemble (par exemple `delta` et `ranked_delta`, "
            "ou `delta` et la `distance aux pairs`)."
        )
        s2_c1, s2_c2 = st.columns(2)
        other_feats = [f for f in inspector.feature_names if f != target_feat]
        default_idx = 0 if other_feats else 0
        with s2_c1:
            feat_x = target_feat
            st.text(f"Axe X : {feat_x}")
        with s2_c2:
            feat_y = st.selectbox(
                "Axe Y :",
                options=other_feats if other_feats else [target_feat],
                index=default_idx,
            )

        if feat_x != feat_y:
            surf_res = inspector.compute_sensitivity_surface(
                feat1=feat_x,
                feat2=feat_y,
                base_features=st.session_state.features_dict,
                grid_size=25,
            )

            fig_surface = go.Figure(
                data=[
                    go.Contour(
                        x=surf_res.grid_x,
                        y=surf_res.grid_y,
                        z=surf_res.scores_grid,
                        colorscale="RdBu",
                        reversescale=True,
                        contours=dict(showlabels=True),
                        colorbar=dict(title="Score"),
                    )
                ]
            )
            # Add current point
            fig_surface.add_trace(
                go.Scatter(
                    x=[surf_res.current_point[0]],
                    y=[surf_res.current_point[1]],
                    mode="markers",
                    name="Point Actuel",
                    marker=dict(color="yellow", size=10, symbol="cross", line=dict(color="black", width=2)),
                )
            )
            fig_surface.update_layout(
                title=f"Carte de Score 2D : S({feat_x}, {feat_y})",
                xaxis_title=feat_x,
                yaxis_title=feat_y,
                height=450,
                template="plotly_white",
            )
            st.plotly_chart(fig_surface, use_container_width=True)


# ============================================================================
# TAB 3: Test en Conditions Réelles (NK Landscape & Trajectoire)
# ============================================================================

with tab_real_nk:
    st.markdown("### 🌐 Test en Conditions Réelles sur Paysage NK")
    st.caption(
        "Ce module instancie un vrai problème NK Landscape avec calcul exact de la fitness et de tous les deltas de voisinage. "
        "Observez le choix du modèle, effectuez des déplacements interactifs (automatiques selon le réseau ou manuels par l'utilisateur) "
        "et suivez l'évolution de la trajectoire."
    )

    # Initialize or recreate NKSession if needed
    if (
        st.session_state.nk_session is None
        or st.session_state.nk_session.n != nk_n
        or st.session_state.nk_session.k != nk_k
        or st.session_state.nk_session.n_agents != n_agents
        or st.session_state.nk_session.seed != nk_seed
    ):
        st.session_state.nk_session = RealNKSession(
            strategy=inspector.strategy,
            n=nk_n,
            k=nk_k,
            interaction_type=nk_topology,
            n_agents=n_agents,
            seed=nk_seed,
        )

    nk_session: RealNKSession = st.session_state.nk_session
    diag = nk_session.evaluate_current_state()

    # Active Agent Selector if Multi-Agent
    active_agent = 0
    if n_agents > 1:
        st.markdown("##### 👥 Sélection de l'Agent Actif")
        agent_labels = [
            f"Agent {i} (Fitness: {diag['agents'][i]['current_fitness']:.4f}, Meilleure: {diag['agents'][i]['best_fitness']:.4f})"
            for i in range(n_agents)
        ]
        active_agent = st.selectbox(
            "Agent à inspecter et déplacer :",
            options=list(range(n_agents)),
            format_func=lambda idx: agent_labels[idx],
            index=min(st.session_state.selected_agent_idx, n_agents - 1),
        )
        st.session_state.selected_agent_idx = active_agent

    agent_data = diag["agents"][active_agent]

    # Current Coordinates & Status Display
    cur_fit = agent_data["current_fitness"]
    best_fit = agent_data["best_fitness"]
    chosen_action = agent_data["chosen_action"]
    chosen_delta = agent_data["chosen_delta"]

    st_c1, st_c2, st_c3, st_c4 = st.columns(4)
    with st_c1:
        st.metric("Étape (Step)", f"{nk_session.step_count}")
    with st_c2:
        st.metric("Fitness Actuelle f(x)", f"{cur_fit:.4f}", delta=f"{chosen_delta:+.4f} (si choix)")
    with st_c3:
        st.metric("Meilleure Fitness f_best", f"{best_fit:.4f}")
    with st_c4:
        if n_agents > 1:
            st.metric("Diversité du Groupe (Hamming)", f"{diag['diversity']:.2f}")
        else:
            st.metric("Stagnation (Sans Progrès)", f"{agent_data['stagnation']}")

    # Bitstring Visualizer & Interactive Bits
    st.markdown("##### 🧬 Coordonnées Réelles de l'Agent (Bitstring)")
    bitstring = agent_data["current_state"]
    bits_html = "".join(
        f'<span class="bit-badge-1">{b}</span>' if b == 1 else f'<span class="bit-badge-0">{b}</span>'
        for b in bitstring
    )
    st.markdown(f'<div style="margin-bottom: 0.8rem;">{bits_html}</div>', unsafe_allow_html=True)

    # Interactive Move Control Buttons
    st.markdown("##### 🎮 Déplacements & Transition de Recherche")
    act_c1, act_c2, act_c3, act_c4 = st.columns([2.5, 2.5, 2.0, 2.0])

    with act_c1:
        btn_label = f"🚀 Appliquer Choix Réseau (Bit {chosen_action}, Δ={chosen_delta:+.3f})"
        if st.button(btn_label, use_container_width=True, type="primary"):
            nk_session.step_model(active_agent)
            st.rerun()

    with act_c2:
        manual_bit = st.selectbox(
            "Bit Manuel à Inverser :",
            options=list(range(nk_n)),
            index=chosen_action,
            key="sel_manual_bit",
            label_visibility="collapsed",
        )
        if st.button(f"👆 Appliquer Déplacement Manuel (Bit {manual_bit})", use_container_width=True):
            nk_session.step_manual(manual_bit, active_agent)
            st.rerun()

    with act_c3:
        if n_agents > 1:
            if st.button("🚀 Avancer TOUS les Agents", use_container_width=True):
                nk_session.step_all_agents_model()
                st.rerun()
        else:
            if st.button("🎲 Redémarrage Aléatoire", use_container_width=True):
                nk_session.reset()
                st.rerun()

    with act_c4:
        if st.button("📋 Copier vers Sliders (Tab 1)", use_container_width=True, help="Transfère les features exactes de l'action choisie dans les contrôles du Tab 1"):
            extracted = agent_data["candidate_actions"][chosen_action]["features"]
            st.session_state.features_dict.update(extracted)
            st.success("✓ Features de l'action transférées dans le Tab 1 !")

    st.markdown("---")

    # Candidate Moves Neighborhood Table & Probabilities
    st.markdown("#### 🎯 Voisinage des 1-Bit Flips Possibles & Choix de la Politique")
    cand_actions = agent_data["candidate_actions"]

    bits = [ca["bit_index"] for ca in cand_actions]
    deltas = [ca["delta_fitness"] for ca in cand_actions]
    logits = [ca["logit"] for ca in cand_actions]
    probs = [ca["probability"] for ca in cand_actions]
    is_impr = [ca["is_improving"] for ca in cand_actions]

    # Bar chart of candidate actions
    bar_colors = ["#10B981" if bit == chosen_action else ("#60A5FA" if d > 0 else "#F87171") for bit, d in zip(bits, deltas, strict=False)]

    fig_actions = go.Figure()
    fig_actions.add_trace(
        go.Bar(
            x=[f"Bit {b}" for b in bits],
            y=logits,
            marker_color=bar_colors,
            text=[f"P={p:.1%}<br>Δ={d:+.3f}" for p, d in zip(probs, deltas, strict=False)],
            textposition="auto",
            hovertemplate="<b>Bit %{x}</b><br>Score / Logit: %{y:+.3f}<extra></extra>",
        )
    )
    fig_actions.update_layout(
        title=f"Distribution des Scores (Logits) pour les {nk_n} Voisins (Vert = Choix du Réseau)",
        xaxis_title="Action Candidate (1-Bit Flip)",
        yaxis_title="Score / Logit Calculé",
        template="plotly_white",
        height=350,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    st.plotly_chart(fig_actions, use_container_width=True)

    # Trajectory Plot
    st.markdown("#### 📈 Historique de la Trajectoire de Recherche")
    history = nk_session.trajectory_history
    steps = [h["step"] for h in history]

    fig_traj = go.Figure()
    if n_agents == 1:
        fits = [h["fitness"][0] for h in history]
        best_fits = [h["best_fitness"][0] for h in history]
        fig_traj.add_trace(go.Scatter(x=steps, y=fits, mode="lines+markers", name="Fitness f(x)", line=dict(color="#2563EB", width=2)))
        fig_traj.add_trace(go.Scatter(x=steps, y=best_fits, mode="lines", name="Meilleure Historique", line=dict(color="#10B981", width=2, dash="dash")))
    else:
        for a in range(n_agents):
            a_fits = [h["fitness"][a] for h in history]
            fig_traj.add_trace(go.Scatter(x=steps, y=a_fits, mode="lines", name=f"Agent {a}"))

    fig_traj.update_layout(
        title="Évolution de la Fitness au Fil des Déplacements",
        xaxis_title="Étape (Step)",
        yaxis_title="Fitness f(x)",
        template="plotly_white",
        height=320,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    st.plotly_chart(fig_traj, use_container_width=True)
