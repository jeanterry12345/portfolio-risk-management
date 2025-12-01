# -*- coding: utf-8 -*-
"""
Marginal VaR et Component VaR
=============================
Décomposition du risque d'un portefeuille multi-actifs.

Référence: Hull, Chapitre 22

Concepts clés:
    - Marginal VaR: Impact d'une petite augmentation de position
    - Component VaR: Contribution de chaque actif au VaR total
    - La somme des Component VaR = VaR total (propriété d'Euler)
"""

import numpy as np
from scipy.stats import norm


def marginal_var(returns_matrix, weights, confidence=0.95, delta=0.01):
    """
    Calcule le Marginal VaR de chaque actif.

    Le Marginal VaR mesure la sensibilité du VaR du portefeuille
    à une petite augmentation du poids de chaque actif.

    Méthode:
        Marginal VaR_i = ∂(VaR) / ∂(w_i)

    Paramètres:
    -----------
    returns_matrix : np.array
        Matrice des rendements (T observations x N actifs)
    weights : np.array
        Poids de chaque actif
    confidence : float
        Niveau de confiance
    delta : float
        Petite variation pour calcul numérique

    Retourne:
    ---------
    np.array : Marginal VaR pour chaque actif

    Exemple:
    --------
    >>> mvars = marginal_var(returns, weights, 0.95)
    >>> print(f"Marginal VaR actif 1: {mvars[0]:.4f}")
    """
    n_assets = returns_matrix.shape[1]
    marginal_vars = np.zeros(n_assets)

    # VaR actuel du portefeuille
    portfolio_returns = np.dot(returns_matrix, weights)
    var_current = _calculate_var_parametric(portfolio_returns, confidence)

    for i in range(n_assets):
        # Augmenter légèrement le poids de l'actif i
        weights_up = weights.copy()
        weights_up[i] += delta
        weights_up = weights_up / weights_up.sum()  # Renormaliser

        # Nouveau VaR
        portfolio_returns_up = np.dot(returns_matrix, weights_up)
        var_up = _calculate_var_parametric(portfolio_returns_up, confidence)

        # Marginal VaR (dérivée numérique)
        marginal_vars[i] = (var_up - var_current) / delta

    return marginal_vars


def component_var(returns_matrix, weights, confidence=0.95):
    """
    Calcule le Component VaR de chaque actif.

    Le Component VaR mesure la contribution de chaque actif
    au VaR total du portefeuille.

    Propriété importante (théorème d'Euler):
        Somme des Component VaR = VaR total

    Formule:
        Component VaR_i = w_i * Marginal VaR_i

    Paramètres:
    -----------
    returns_matrix : np.array
        Matrice des rendements (T x N)
    weights : np.array
        Poids de chaque actif
    confidence : float
        Niveau de confiance

    Retourne:
    ---------
    dict : contenant component_var, percent_contribution, total_var

    Exemple:
    --------
    >>> result = component_var(returns, weights, 0.95)
    >>> print(f"Contribution actif 1: {result['percent_contribution'][0]:.1f}%")
    """
    # Marginal VaR
    mvar = marginal_var(returns_matrix, weights, confidence)

    # Component VaR = poids * Marginal VaR
    cvar = weights * mvar

    # VaR total (somme des component VaR par théorème d'Euler)
    total_var = np.sum(cvar)

    # Contribution en pourcentage
    percent_contrib = (cvar / total_var) * 100 if total_var > 0 else np.zeros_like(cvar)

    return {
        'component_var': cvar,
        'percent_contribution': percent_contrib,
        'marginal_var': mvar,
        'total_var': total_var
    }


def component_var_analytical(cov_matrix, weights, confidence=0.95):
    """
    Calcule le Component VaR de manière analytique.

    Formule analytique (sous hypothèse normale):
        Component VaR_i = w_i * (Σ * w)_i / σ_p * VaR_p

    où Σ est la matrice de covariance et σ_p l'écart-type du portefeuille.

    Paramètres:
    -----------
    cov_matrix : np.array
        Matrice de covariance des rendements (N x N)
    weights : np.array
        Poids de chaque actif
    confidence : float
        Niveau de confiance

    Retourne:
    ---------
    dict : Component VaR analytique
    """
    # Variance du portefeuille
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_var)

    # Quantile normal
    z_alpha = norm.ppf(confidence)

    # VaR du portefeuille
    var_portfolio = portfolio_std * z_alpha

    # Contribution de chaque actif à la variance
    # (Σ * w)_i représente la covariance de l'actif i avec le portefeuille
    marginal_contrib = np.dot(cov_matrix, weights)

    # Component VaR = w_i * cov(i, portfolio) / σ_p * z_α
    component = weights * marginal_contrib / portfolio_std * z_alpha

    # Normalisation (la somme doit égaler le VaR total)
    total = np.sum(component)
    percent_contrib = (component / total) * 100 if total > 0 else np.zeros_like(component)

    return {
        'component_var': component,
        'percent_contribution': percent_contrib,
        'total_var': var_portfolio,
        'portfolio_std': portfolio_std
    }


def risk_decomposition_report(asset_names, returns_matrix, weights, confidence=0.95):
    """
    Génère un rapport de décomposition du risque.

    Paramètres:
    -----------
    asset_names : list
        Noms des actifs
    returns_matrix : np.array
        Matrice des rendements
    weights : np.array
        Poids du portefeuille
    confidence : float
        Niveau de confiance

    Retourne:
    ---------
    str : Rapport formaté
    """
    result = component_var(returns_matrix, weights, confidence)

    lines = [
        f"\n{'='*60}",
        f"DÉCOMPOSITION DU RISQUE - VaR {confidence*100:.0f}%",
        f"{'='*60}",
        f"\nVaR Total: {result['total_var']*100:.4f}%",
        f"\n{'Actif':<15} {'Poids':<10} {'Marg. VaR':<12} {'Comp. VaR':<12} {'Contrib.':<10}",
        f"{'-'*60}"
    ]

    for i, name in enumerate(asset_names):
        lines.append(
            f"{name:<15} {weights[i]*100:>8.2f}% "
            f"{result['marginal_var'][i]*100:>10.4f}% "
            f"{result['component_var'][i]*100:>10.4f}% "
            f"{result['percent_contribution'][i]:>8.1f}%"
        )

    lines.append(f"{'-'*60}")
    lines.append(f"{'TOTAL':<15} {sum(weights)*100:>8.2f}% "
                 f"{'':>12} "
                 f"{result['total_var']*100:>10.4f}% "
                 f"{'100.0':>8}%")

    return '\n'.join(lines)


def _calculate_var_parametric(returns, confidence):
    """
    Fonction utilitaire pour calculer le VaR paramétrique.

    Paramètres:
    -----------
    returns : np.array
        Rendements
    confidence : float
        Niveau de confiance

    Retourne:
    ---------
    float : VaR
    """
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    z_alpha = norm.ppf(confidence)

    return -mean + std * z_alpha


# Tests si exécuté directement
if __name__ == "__main__":
    # Générer des données de test pour 3 actifs
    np.random.seed(42)
    n_days = 252
    n_assets = 3

    # Corrélations entre actifs
    corr_matrix = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])

    # Volatilités individuelles
    vols = np.array([0.15, 0.20, 0.25])  # 15%, 20%, 25%

    # Matrice de covariance
    cov_matrix = np.outer(vols, vols) * corr_matrix

    # Générer des rendements corrélés
    L = np.linalg.cholesky(cov_matrix)
    returns_uncorr = np.random.normal(0, 1, (n_days, n_assets))
    returns_matrix = np.dot(returns_uncorr, L.T) / np.sqrt(252)  # Rendements journaliers

    # Poids du portefeuille
    weights = np.array([0.4, 0.35, 0.25])  # 40%, 35%, 25%
    asset_names = ['Actif A', 'Actif B', 'Actif C']

    print("=== Test Marginal et Component VaR ===")

    # Rapport complet
    report = risk_decomposition_report(asset_names, returns_matrix, weights, 0.95)
    print(report)

    # Comparaison avec méthode analytique
    print("\n--- Comparaison numérique vs analytique ---")
    result_num = component_var(returns_matrix, weights, 0.95)
    result_ana = component_var_analytical(cov_matrix / 252, weights, 0.95)

    print(f"\nVaR numérique:   {result_num['total_var']*100:.4f}%")
    print(f"VaR analytique:  {result_ana['total_var']*100:.4f}%")
