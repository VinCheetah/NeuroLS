# NeuroLS: Local Search & Neuroevolution Framework

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

**NeuroLS** est un framework modulaire, hautement typé et optimisé pour la recherche locale (Local Search) et la neuroévolution (notamment via **CMA-ES**). Il permet d'étudier des heuristiques classiques, des métaheuristiques et des politiques neuronales, aussi bien en configuration **mono-agent** qu'en **groupes multi-agents** collaboratifs ou compétitifs.

---

## 🌟 Points Clés & Fonctionnalités

1. **Typage strict & Architecture Orientée Objet / Générique** :
   - Typage moderne (PEP 585/604/695), protocoles et classes de base abstraites (`BaseProblem`, `BaseStrategy`, `BaseOptimizer`).
   - Contrats de déterminisme explicites (`is_deterministic`, mode `state_dependent_deterministic`).
   - Spécification formelle des hyperparamètres (`HyperparameterSpec`) avec tuners automatiques (Grid, Random, CMA-ES).

2. **Calculs par Batch Vectorisés Très Optimisés (PyTorch)** :
   - Dimensions de batch flexibles : `(n_instances, n_restarts, n_agents)` ou aplaties `(B, ...)`.
   - Évaluation simultanée de milliers d'agents et de tous les voisins en quelques millisecondes sur CPU ou GPU.
   - Mode traditionnel pas-à-pas (`TraditionalRunner`) pour le débogage et l'inspection unitaire.

3. **Multi-Agent & Partage d'Informations Modulable** :
   - Gestion de groupes d'agents avec partage configurable : distances de Hamming relatives, classements (ranks), consensus de variables, scores et écart au meilleur du groupe.
   - Suivi dynamique de la diversité de population au cours de la recherche.

4. **Problèmes de Recherche Locale Extensibles** :
   - **NK Landscapes** (problème par défaut) : topologie adjacente ou aléatoire, calcul incrémental et vectorisé des $\Delta f$.
   - **MultiNKLandscape** : plusieurs instances de paysages évaluées en parallèle.
   - **OneMax** : exemple linéaire démontrant la généralisation immédiate à d'autres problèmes discrets ou continus.

5. **Stratégies de Recherche Complètes** :
   - **Classiques** : `HillClimber` (Steepest Ascent), `FirstImprovement`, `RandomWalk`, `SimulatedAnnealing`, `TabuSearch` (avec mémoire taboue tensorielle et critère d'aspiration).
   - **Neuronales** : `NeuralLocalSearchStrategy` avec canaux d'entrées riches ($\Delta f$, indicateur d'amélioration, bit actuel, contexte de trajectoire, consensus, rang, distances de groupe).
   - **Architectures** : `LinearScorer` (100% interprétable), `EquivariantMLPPolicy`, `DeepSetsPolicy`.

6. **Neuroévolution Propre & Optimisée (CMA-ES)** :
   - Optimisation des paramètres de politiques neuronales par `CMAESOptimizer`.
   - Rollouts vectorisés et Common Random Numbers (CRN) pour minimiser la variance d'évaluation.
   - Checkpointing automatique et traçabilité complète des générations.

7. **Visualisation & Outils d'Interprétabilité pour Réseaux de Neurones** :
   - Visualisation de l'importance des caractéristiques (poids linéaires et sensibilités de gradient).
   - Explication pas-à-pas de chaque décision prise par le réseau.
   - Courbes de convergence, boxplots comparatifs et dynamiques multi-agents prêts pour publication.

8. **CLI Complète (Rich & Typer)** :
   - Barres de progression en direct, tableaux de classement interactifs et commandes dédiées : `run`, `benchmark`, `train`, `inspect-nn`, `inspect-nk`.

---

## 📁 Structure du Projet

```text
src/neurols/
├── types.py                  # Protocoles, Enums, TypedDicts, generics (PEP 585/604)
├── core/
│   ├── problem.py            # BaseProblem (ABC), BatchedProblem
│   ├── group.py              # GroupConfig, GroupContext, GroupMetrics
│   ├── trajectory.py         # Trajectory, StepRecord, synthèses
│   └── runner.py             # BatchedRunner (haute vitesse) & TraditionalRunner
├── problems/
│   ├── nk_landscape.py       # NKLandscape, MultiNKLandscape vectorisé en PyTorch
│   └── onemax.py             # OneMax (problème linéaire de validation)
├── strategies/
│   ├── base.py               # BaseStrategy, contrat de déterminisme
│   ├── classical/            # HillClimber, FirstImprovement, RandomWalk, SA, TabuSearch
│   ├── neural/               # Features, LinearScorer, MLP, DeepSets, NeuralStrategy
│   └── tuning.py             # HyperparameterTuner (GridSearch, RandomSearch)
├── training/
│   ├── base.py               # BaseOptimizer, TrainResult, Checkpoints
│   └── cma_es.py             # CMAESOptimizer pour neuroévolution
├── analysis/
│   ├── metrics.py            # AUC, temps d'atteinte, taux de succès
│   ├── experiment.py         # ExperimentLogger, comparaison multi-méthodes
│   ├── plotting.py           # Graphiques matplotlib publication-grade
│   └── interpretability.py   # Salience, sensibilité et explication de décision
└── cli/
    └── app.py                # Commandes Typer avec affichage Rich
```

---

## 🚀 Installation

Avec `uv` :
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Pour lancer les tests et vérifier le linting :
```bash
pytest
ruff check .
ruff format --check .
```

---

## 💻 Utilisation de la CLI

### 1. Analyser les propriétés d'un Paysage NK
Estimer la rugosité, la densité des optima locaux et le nombre de mouvements améliorants :
```bash
neurols inspect-nk -n 20 -k 4 --topology adjacent --samples 2000
```

### 2. Exécuter une recherche locale (Mono ou Multi-Agent)
Lancer une recherche avec affichage interactif de la progression et métriques finales :
```bash
# Mono-agent
neurols run --strategy HillClimber -n 25 -k 3 --steps 100 --restarts 8

# Groupe Multi-Agent (4 agents par groupe s'échangeant des informations)
neurols run --strategy TabuSearch -n 25 -k 4 --steps 100 --restarts 4 --agents 4 --share-info
```

### 3. Comparer plusieurs stratégies (Benchmark)
Compare `HillClimber`, `FirstImprovement`, `TabuSearch`, `SimulatedAnnealing` et `NeuralStrategy` sur le même problème avec les mêmes germes aléatoires, et génère des graphiques comparatifs :
```bash
neurols benchmark -n 25 -k 3 --steps 100 --restarts 16 --plot --out experiments/benchmark
```

### 4. Entraîner une politique neuronale avec CMA-ES
Optimise les poids d'un réseau de neurones sur des paysages NK :
```bash
neurols train -n 20 -k 3 -g 15 --popsize 12 --eval-restarts 8 --eval-steps 40 --policy-type linear --plot --ckpt-dir checkpoints/
```

### 5. Inspecter et comprendre les décisions du Réseau de Neurones
Visualiser les poids accordés à chaque canal d'entrée ($\Delta f$, distances, consensus, stagnation) et expliquer un coup :
```bash
neurols inspect-nn --ckpt checkpoints/cma_checkpoint_gen15.json -n 20 -k 3 --out diagnostics/
```

---

## 🛠️ Guide d'Extension

### Ajouter un Nouveau Problème
Héritez de `BatchedProblem` et implémentez les méthodes requises :

```python
import torch
from neurols.core.problem import BatchedProblem


class MonProbleme(BatchedProblem):
    def __init__(self, dimension: int, device: str = "cpu"):
        super().__init__(device=device)
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def neighborhood_size(self) -> int:
        return self._dim

    def evaluate(self, states: torch.Tensor) -> torch.Tensor:
        # states: (B, dimension) -> renvoie fitness: (B,)
        return states.float().sum(dim=-1)

    def sample_initial_states(self, batch_shape, generator=None) -> torch.Tensor:
        return torch.randint(0, 2, (*batch_shape, self._dim), dtype=torch.int8)

    def get_neighborhood_deltas(self, states: torch.Tensor) -> torch.Tensor:
        # Renvoie les deltas pour chaque voisin: (B, neighborhood_size)
        return ...

    def apply_moves(self, states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        # Applique le mouvement choisi: moves: (B,)
        return ...

    def compute_distances(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return (s1 != s2).sum(dim=-1).float()
```

### Ajouter une Nouvelle Stratégie
Héritez de `BaseStrategy` et définissez vos hyperparamètres et la méthode `select_move` :

```python
from neurols.strategies.base import BaseStrategy
from neurols.types import HyperparameterSpec, MoveResult


class MaStrategie(BaseStrategy):
    def __init__(self, mon_param: float = 0.5):
        super().__init__(name="MaStrategie")
        self.mon_param = mon_param

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def hyperparameter_specs(self) -> list[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="mon_param",
                param_type="float",
                default=0.5,
                low=0.0,
                high=1.0,
                description="Facteur d'exploration.",
            )
        ]

    @property
    def hyperparameters(self) -> dict:
        return {"mon_param": self.mon_param}

    def reset(self, batch_size, device, seed=None):
        super().reset(batch_size=batch_size, device=device, seed=seed)

    def select_move(
        self,
        states,
        deltas,
        current_fitness,
        best_fitness,
        group_context=None,
        group_metrics=None,
        step=0,
    ):
        # Logique de sélection vectorisée
        chosen = torch.argmax(deltas, dim=-1)
        is_improving = deltas.gather(1, chosen.unsqueeze(1)).squeeze(1) > 0
        return MoveResult(chosen_moves=chosen, is_improving=is_improving)
```

---

## 🔬 Licence & Contribution

Projet développé avec passion pour la recherche en optimisation combinatoire et métaheuristiques assistées par intelligence artificielle.
Contributions et pull requests bienvenues !
