# Plateforme de Gestion du Risque de Portefeuille

## Description

Plateforme complète pour la mesure et la gestion du risque de portefeuille, incluant VaR, CVaR, GARCH et backtesting.

**Niveau** : M2 Finance Quantitative - Sorbonne Université

## Fonctionnalités

- **VaR (Value at Risk)** : Trois méthodes
  - Historique (simulation historique)
  - Variance-covariance (paramétrique)
  - Monte Carlo (simulation)
- **CVaR / Expected Shortfall** : Mesure de risque cohérente
- **Marginal VaR** : Impact d'une position supplémentaire
- **Component VaR** : Contribution de chaque actif au VaR total
- **GARCH(1,1)** : Estimation de la volatilité conditionnelle
- **Stress Testing** : Scénarios 2008, COVID-19
- **Backtesting** : Tests de Kupiec et Christoffersen
- **Reporting Excel** : Export automatisé des résultats

## Structure du Projet

```
portfolio-risk-management/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── var_calculator.py     # Calcul du VaR (3 méthodes)
│   ├── cvar_calculator.py    # CVaR et mesures de risque
│   ├── marginal_component.py # Marginal VaR et Component VaR
│   ├── garch_model.py        # Modèle GARCH(1,1)
│   ├── stress_testing.py     # Scénarios de stress
│   ├── backtesting.py        # Tests de Kupiec et Christoffersen
│   └── reporting.py          # Export Excel
├── tests/
│   └── test_var.py           # Tests unitaires
└── examples/
    └── demo.py               # Démonstration avec données CAC40
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation Rapide

```python
from src.var_calculator import VaRCalculator

# Créer le calculateur avec des rendements
calculator = VaRCalculator(returns)

# VaR à 95% sur 1 jour
var_hist = calculator.var_historical(confidence=0.95)
var_param = calculator.var_parametric(confidence=0.95)
var_mc = calculator.var_monte_carlo(confidence=0.95)

print(f"VaR Historique: {var_hist:.2%}")
print(f"VaR Paramétrique: {var_param:.2%}")
print(f"VaR Monte Carlo: {var_mc:.2%}")
```

## Références Théoriques

- Hull, J.C. (2018). *Options, Futures, and Other Derivatives*, 9th Edition
  - Chapitre 22 : Value at Risk
  - Chapitre 23 : Estimating Volatilities and Correlations (GARCH)

## Résultats

- Couverture VaR 95% validée par backtesting
- Tests de Kupiec et Christoffersen passés

## Auteur

Projet M2 Finance Quantitative - Sorbonne Université
