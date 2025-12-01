# -*- coding: utf-8 -*-
"""
Calcul du Value at Risk (VaR)
=============================
Trois méthodes de calcul du VaR selon Hull, Chapitre 22.

Le VaR répond à la question:
"Quelle est la perte maximale avec X% de confiance sur N jours?"

Méthodes implémentées:
    1. Historique (simulation historique)
    2. Variance-covariance (paramétrique)
    3. Monte Carlo (simulation)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class VaRCalculator:
    """
    Classe pour calculer le Value at Risk selon différentes méthodes.

    Attributs:
    ----------
    returns : np.array
        Série des rendements historiques
    portfolio_value : float
        Valeur du portefeuille (pour exprimer le VaR en €)

    Exemple:
    --------
    >>> calculator = VaRCalculator(returns, portfolio_value=1000000)
    >>> var_95 = calculator.var_historical(confidence=0.95)
    >>> print(f"VaR 95%: {var_95:.2%}")
    """

    def __init__(self, returns, portfolio_value=1.0):
        """
        Initialise le calculateur de VaR.

        Paramètres:
        -----------
        returns : array-like
            Rendements historiques (en décimal, ex: 0.02 pour 2%)
        portfolio_value : float
            Valeur du portefeuille pour calculs en valeur absolue
        """
        self.returns = np.array(returns)
        self.portfolio_value = portfolio_value

        # Statistiques de base
        self.mean = np.mean(self.returns)
        self.std = np.std(self.returns, ddof=1)  # Écart-type non biaisé
        self.n_obs = len(self.returns)

    def var_historical(self, confidence=0.95, horizon=1):
        """
        Calcule le VaR par la méthode historique.

        Méthode (Hull, Section 22.1):
            - Trier les rendements historiques
            - Le VaR est le (1-confidence) quantile

        Paramètres:
        -----------
        confidence : float
            Niveau de confiance (ex: 0.95 pour 95%)
        horizon : int
            Horizon en jours (hypothèse: √T pour ajustement)

        Retourne:
        ---------
        float : VaR en pourcentage (toujours positif)

        Note:
            Le VaR est exprimé comme une perte, donc positif.
            Un VaR de 2% signifie qu'il y a (1-confidence)% de chances
            de perdre plus de 2%.
        """
        # Quantile correspondant à la confiance
        # Ex: pour 95%, on prend le 5ème percentile
        alpha = 1 - confidence
        var_1day = -np.percentile(self.returns, alpha * 100)

        # Ajustement pour l'horizon (hypothèse racine du temps)
        var_horizon = var_1day * np.sqrt(horizon)

        return var_horizon

    def var_parametric(self, confidence=0.95, horizon=1):
        """
        Calcule le VaR par la méthode variance-covariance (paramétrique).

        Méthode (Hull, Section 22.2):
            - Suppose que les rendements suivent une loi normale
            - VaR = -μ + σ * z_α

        où z_α est le quantile de la loi normale standard.

        Paramètres:
        -----------
        confidence : float
            Niveau de confiance
        horizon : int
            Horizon en jours

        Retourne:
        ---------
        float : VaR en pourcentage
        """
        # Quantile de la loi normale
        z_alpha = norm.ppf(confidence)

        # VaR sur 1 jour
        # Note: on utilise -mean car le VaR est une perte
        var_1day = -self.mean + self.std * z_alpha

        # Ajustement pour l'horizon
        # μ se scale linéairement, σ en racine du temps
        var_horizon = -self.mean * horizon + self.std * np.sqrt(horizon) * z_alpha

        return var_horizon

    def var_monte_carlo(self, confidence=0.95, horizon=1, n_simulations=10000,
                         seed=None):
        """
        Calcule le VaR par simulation Monte Carlo.

        Méthode (Hull, Section 22.5):
            1. Simuler N scénarios de rendements
            2. Calculer le P&L pour chaque scénario
            3. Le VaR est le (1-confidence) quantile des pertes

        Paramètres:
        -----------
        confidence : float
            Niveau de confiance
        horizon : int
            Horizon en jours
        n_simulations : int
            Nombre de simulations
        seed : int
            Graine aléatoire pour reproductibilité

        Retourne:
        ---------
        float : VaR en pourcentage
        """
        if seed is not None:
            np.random.seed(seed)

        # Simuler des rendements normaux
        # On utilise les paramètres estimés (mean, std)
        simulated_returns = np.random.normal(
            self.mean * horizon,
            self.std * np.sqrt(horizon),
            n_simulations
        )

        # VaR = quantile négatif (car VaR = perte)
        alpha = 1 - confidence
        var = -np.percentile(simulated_returns, alpha * 100)

        return var

    def var_in_value(self, var_percent):
        """
        Convertit le VaR en pourcentage vers une valeur absolue.

        Paramètres:
        -----------
        var_percent : float
            VaR en pourcentage

        Retourne:
        ---------
        float : VaR en valeur absolue (€)
        """
        return var_percent * self.portfolio_value

    def calculate_all(self, confidence=0.95, horizon=1):
        """
        Calcule le VaR avec les trois méthodes.

        Retourne:
        ---------
        dict : VaR calculé par chaque méthode
        """
        return {
            'historical': self.var_historical(confidence, horizon),
            'parametric': self.var_parametric(confidence, horizon),
            'monte_carlo': self.var_monte_carlo(confidence, horizon)
        }


def var_portfolio(returns_matrix, weights, confidence=0.95, method='parametric'):
    """
    Calcule le VaR d'un portefeuille multi-actifs.

    Paramètres:
    -----------
    returns_matrix : np.array
        Matrice des rendements (T x N), T observations, N actifs
    weights : np.array
        Poids de chaque actif dans le portefeuille
    confidence : float
        Niveau de confiance
    method : str
        Méthode de calcul ('historical', 'parametric', 'monte_carlo')

    Retourne:
    ---------
    float : VaR du portefeuille
    """
    # Rendements du portefeuille
    portfolio_returns = np.dot(returns_matrix, weights)

    # Créer un calculateur pour le portefeuille
    calculator = VaRCalculator(portfolio_returns)

    if method == 'historical':
        return calculator.var_historical(confidence)
    elif method == 'parametric':
        return calculator.var_parametric(confidence)
    elif method == 'monte_carlo':
        return calculator.var_monte_carlo(confidence)
    else:
        raise ValueError(f"Méthode inconnue: {method}")


# Tests si exécuté directement
if __name__ == "__main__":
    # Générer des rendements aléatoires pour test
    np.random.seed(42)
    n_days = 252  # 1 an de données

    # Rendements normaux avec mean=0.05% par jour et std=1.5%
    returns = np.random.normal(0.0005, 0.015, n_days)

    print("=== Test VaR Calculator ===")
    print(f"Nombre d'observations: {n_days}")
    print(f"Rendement moyen: {np.mean(returns)*100:.4f}%")
    print(f"Écart-type: {np.std(returns)*100:.4f}%")

    # Créer le calculateur
    calculator = VaRCalculator(returns, portfolio_value=1000000)

    # Calculer le VaR à 95% et 99%
    for confidence in [0.95, 0.99]:
        print(f"\n--- VaR {confidence*100:.0f}% ---")
        var_results = calculator.calculate_all(confidence)

        print(f"Historique:   {var_results['historical']*100:.4f}%")
        print(f"Paramétrique: {var_results['parametric']*100:.4f}%")
        print(f"Monte Carlo:  {var_results['monte_carlo']*100:.4f}%")

        # En valeur absolue
        var_value = calculator.var_in_value(var_results['parametric'])
        print(f"VaR paramétrique en €: {var_value:,.0f} €")
