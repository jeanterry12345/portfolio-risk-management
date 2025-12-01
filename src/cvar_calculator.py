# -*- coding: utf-8 -*-
"""
CVaR / Expected Shortfall
=========================
Calcul de la Conditional Value at Risk (CVaR), aussi appelée
Expected Shortfall (ES).

Référence: Hull, Chapitre 22, Section 22.6

Le CVaR répond à la question:
"Quelle est la perte moyenne dans les (1-confidence)% pires cas?"

Avantages du CVaR par rapport au VaR:
    - Mesure de risque cohérente (sous-additivité)
    - Prend en compte la queue de distribution
    - Meilleure gestion du risque extrême
"""

import numpy as np
from scipy.stats import norm


def calculate_cvar(returns, confidence=0.95):
    """
    Calcule le CVaR (Expected Shortfall) par la méthode historique.

    Le CVaR est la moyenne des pertes qui dépassent le VaR.

    Méthode:
        1. Trouver le VaR au niveau de confiance donné
        2. Calculer la moyenne des rendements inférieurs au -VaR

    Paramètres:
    -----------
    returns : array-like
        Rendements historiques
    confidence : float
        Niveau de confiance (ex: 0.95)

    Retourne:
    ---------
    float : CVaR en pourcentage (positif = perte)

    Exemple:
    --------
    >>> returns = np.random.normal(0, 0.02, 252)
    >>> cvar = calculate_cvar(returns, confidence=0.95)
    >>> print(f"CVaR 95%: {cvar:.2%}")
    """
    returns = np.array(returns)

    # Seuil: (1-confidence) quantile des rendements
    alpha = 1 - confidence
    var_threshold = np.percentile(returns, alpha * 100)

    # CVaR = moyenne des rendements en dessous du seuil
    # (les pertes extrêmes)
    tail_losses = returns[returns <= var_threshold]

    if len(tail_losses) == 0:
        # Pas assez de données
        return -var_threshold

    # CVaR est exprimé comme une perte (positif)
    cvar = -np.mean(tail_losses)

    return cvar


def calculate_expected_shortfall(returns, confidence=0.95):
    """
    Alias pour calculate_cvar.

    CVaR et Expected Shortfall sont deux noms pour la même mesure.
    """
    return calculate_cvar(returns, confidence)


def cvar_parametric(mean, std, confidence=0.95):
    """
    Calcule le CVaR paramétrique (hypothèse de normalité).

    Formule (sous hypothèse normale):
        CVaR = -μ + σ * φ(z_α) / (1-α)

    où φ est la densité normale et z_α est le quantile.

    Paramètres:
    -----------
    mean : float
        Moyenne des rendements
    std : float
        Écart-type des rendements
    confidence : float
        Niveau de confiance

    Retourne:
    ---------
    float : CVaR paramétrique
    """
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)

    # Densité normale au point z_alpha
    phi_z = norm.pdf(z_alpha)

    # CVaR paramétrique
    cvar = -mean + std * phi_z / alpha

    return cvar


def var_cvar_comparison(returns, confidence=0.95):
    """
    Compare le VaR et le CVaR.

    Le CVaR est toujours >= VaR car il prend en compte
    toute la queue de distribution.

    Paramètres:
    -----------
    returns : array-like
        Rendements historiques
    confidence : float
        Niveau de confiance

    Retourne:
    ---------
    dict : VaR, CVaR et ratio CVaR/VaR
    """
    returns = np.array(returns)

    # VaR historique
    alpha = 1 - confidence
    var = -np.percentile(returns, alpha * 100)

    # CVaR
    cvar = calculate_cvar(returns, confidence)

    # Ratio CVaR/VaR (mesure de la queue de distribution)
    ratio = cvar / var if var > 0 else np.nan

    return {
        'var': var,
        'cvar': cvar,
        'ratio': ratio,
        'excess': cvar - var  # Combien le CVaR dépasse le VaR
    }


def marginal_cvar(returns_matrix, weights, confidence=0.95, asset_index=0,
                   delta=0.01):
    """
    Calcule le Marginal CVaR d'un actif dans le portefeuille.

    Le Marginal CVaR mesure comment le CVaR change quand on
    augmente légèrement le poids d'un actif.

    Paramètres:
    -----------
    returns_matrix : np.array
        Matrice des rendements (T x N)
    weights : np.array
        Poids actuels
    confidence : float
        Niveau de confiance
    asset_index : int
        Index de l'actif à analyser
    delta : float
        Petite variation du poids

    Retourne:
    ---------
    float : Marginal CVaR
    """
    # CVaR actuel
    portfolio_returns = np.dot(returns_matrix, weights)
    cvar_current = calculate_cvar(portfolio_returns, confidence)

    # Nouveaux poids avec delta sur l'actif
    weights_new = weights.copy()
    weights_new[asset_index] += delta
    # Renormaliser pour que la somme = 1
    weights_new = weights_new / weights_new.sum()

    # Nouveau CVaR
    portfolio_returns_new = np.dot(returns_matrix, weights_new)
    cvar_new = calculate_cvar(portfolio_returns_new, confidence)

    # Marginal CVaR
    marginal = (cvar_new - cvar_current) / delta

    return marginal


# Tests si exécuté directement
if __name__ == "__main__":
    # Générer des rendements avec queue épaisse (Student-t)
    np.random.seed(42)
    n_days = 500

    # Rendements normaux
    returns_normal = np.random.normal(0, 0.02, n_days)

    # Rendements avec queue épaisse (mélange normal + extrêmes)
    returns_fat = returns_normal.copy()
    # Ajouter quelques événements extrêmes
    extreme_indices = np.random.choice(n_days, 10, replace=False)
    returns_fat[extreme_indices] = np.random.uniform(-0.10, -0.05, 10)

    print("=== Test CVaR ===")

    # Rendements normaux
    print("\n--- Rendements normaux ---")
    comparison_normal = var_cvar_comparison(returns_normal, 0.95)
    print(f"VaR 95%:  {comparison_normal['var']*100:.4f}%")
    print(f"CVaR 95%: {comparison_normal['cvar']*100:.4f}%")
    print(f"Ratio CVaR/VaR: {comparison_normal['ratio']:.2f}")

    # Rendements avec queue épaisse
    print("\n--- Rendements avec queue épaisse ---")
    comparison_fat = var_cvar_comparison(returns_fat, 0.95)
    print(f"VaR 95%:  {comparison_fat['var']*100:.4f}%")
    print(f"CVaR 95%: {comparison_fat['cvar']*100:.4f}%")
    print(f"Ratio CVaR/VaR: {comparison_fat['ratio']:.2f}")

    # Comparaison CVaR paramétrique vs historique
    print("\n--- Comparaison paramétrique vs historique ---")
    mean = np.mean(returns_normal)
    std = np.std(returns_normal)
    cvar_param = cvar_parametric(mean, std, 0.95)
    cvar_hist = calculate_cvar(returns_normal, 0.95)
    print(f"CVaR paramétrique: {cvar_param*100:.4f}%")
    print(f"CVaR historique:   {cvar_hist*100:.4f}%")
