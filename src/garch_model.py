# -*- coding: utf-8 -*-
"""
Modèle GARCH(1,1)
=================
Estimation de la volatilité conditionnelle avec le modèle GARCH(1,1).

Référence: Hull, Chapitre 23 - Estimating Volatilities and Correlations

Le modèle GARCH(1,1) capture le fait que la volatilité est:
    - Variable dans le temps (hétéroscédasticité)
    - Auto-corrélée (clustering de volatilité)

Modèle GARCH(1,1):
    σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

où:
    - ω > 0 (terme constant)
    - α ≥ 0 (réaction aux chocs)
    - β ≥ 0 (persistance)
    - α + β < 1 (condition de stationnarité)
"""

import numpy as np
from scipy.optimize import minimize


class GARCHModel:
    """
    Implémentation du modèle GARCH(1,1).

    Attributs:
    ----------
    returns : np.array
        Série des rendements
    omega : float
        Paramètre ω (terme constant)
    alpha : float
        Paramètre α (réaction aux chocs)
    beta : float
        Paramètre β (persistance)

    Exemple:
    --------
    >>> model = GARCHModel(returns)
    >>> model.fit()
    >>> vol_forecast = model.forecast(horizon=10)
    """

    def __init__(self, returns):
        """
        Initialise le modèle GARCH.

        Paramètres:
        -----------
        returns : array-like
            Rendements historiques
        """
        self.returns = np.array(returns)
        self.n_obs = len(returns)

        # Paramètres (à estimer)
        self.omega = None
        self.alpha = None
        self.beta = None

        # Volatilités conditionnelles estimées
        self.conditional_vol = None

    def _compute_variance(self, params):
        """
        Calcule les variances conditionnelles pour des paramètres donnés.

        Paramètres:
        -----------
        params : tuple
            (omega, alpha, beta)

        Retourne:
        ---------
        np.array : variances conditionnelles σ²_t
        """
        omega, alpha, beta = params
        n = self.n_obs
        variance = np.zeros(n)

        # Initialisation: variance inconditionnelle
        # E[σ²] = ω / (1 - α - β)
        if (1 - alpha - beta) > 0.001:
            var_uncond = omega / (1 - alpha - beta)
        else:
            var_uncond = np.var(self.returns)

        variance[0] = var_uncond

        # Récurrence GARCH(1,1)
        for t in range(1, n):
            variance[t] = omega + alpha * self.returns[t-1]**2 + beta * variance[t-1]

        return variance

    def _negative_log_likelihood(self, params):
        """
        Calcule la log-vraisemblance négative (à minimiser).

        Sous hypothèse de normalité conditionnelle:
            r_t | I_{t-1} ~ N(0, σ²_t)

        Log-vraisemblance:
            L = -0.5 * Σ [log(σ²_t) + r²_t / σ²_t]

        Paramètres:
        -----------
        params : tuple
            (omega, alpha, beta)

        Retourne:
        ---------
        float : -log(L)
        """
        omega, alpha, beta = params

        # Contraintes
        if omega <= 0 or alpha < 0 or beta < 0:
            return 1e10
        if alpha + beta >= 1:
            return 1e10

        variance = self._compute_variance(params)

        # Éviter log(0) et division par 0
        variance = np.maximum(variance, 1e-10)

        # Log-vraisemblance (sans constante)
        log_likelihood = -0.5 * np.sum(
            np.log(variance) + self.returns**2 / variance
        )

        return -log_likelihood  # On minimise le négatif

    def fit(self):
        """
        Estime les paramètres du modèle GARCH(1,1) par maximum de vraisemblance.

        Retourne:
        ---------
        dict : paramètres estimés et statistiques
        """
        # Variance empirique pour initialisation
        var_emp = np.var(self.returns)

        # Paramètres initiaux
        # Typiquement: α ≈ 0.1, β ≈ 0.85
        alpha_init = 0.10
        beta_init = 0.85
        omega_init = var_emp * (1 - alpha_init - beta_init)

        initial_params = [omega_init, alpha_init, beta_init]

        # Optimisation
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[(1e-8, None), (0, 0.999), (0, 0.999)]
        )

        # Stocker les paramètres
        self.omega = result.x[0]
        self.alpha = result.x[1]
        self.beta = result.x[2]

        # Calculer les volatilités conditionnelles
        variance = self._compute_variance(result.x)
        self.conditional_vol = np.sqrt(variance)

        # Variance inconditionnelle (long-terme)
        if (1 - self.alpha - self.beta) > 0.001:
            long_term_var = self.omega / (1 - self.alpha - self.beta)
        else:
            long_term_var = var_emp

        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'persistence': self.alpha + self.beta,
            'long_term_vol': np.sqrt(long_term_var),
            'log_likelihood': -result.fun,
            'converged': result.success
        }

    def forecast(self, horizon=1):
        """
        Prévision de la volatilité future.

        Formule de prévision GARCH(1,1):
            E[σ²_{t+h}] = VL + (α + β)^h * (σ²_t - VL)

        où VL = ω / (1 - α - β) est la variance long-terme.

        Paramètres:
        -----------
        horizon : int
            Nombre de jours de prévision

        Retourne:
        ---------
        np.array : volatilités prévues
        """
        if self.omega is None:
            raise ValueError("Le modèle doit être estimé d'abord (appeler fit())")

        # Variance long-terme
        persistence = self.alpha + self.beta
        if (1 - persistence) > 0.001:
            var_long_term = self.omega / (1 - persistence)
        else:
            var_long_term = self.conditional_vol[-1]**2

        # Dernière variance observée
        var_current = self.conditional_vol[-1]**2

        # Prévisions
        forecasts = np.zeros(horizon)
        for h in range(horizon):
            forecasts[h] = var_long_term + (persistence**(h+1)) * (var_current - var_long_term)

        # Retourner en volatilité (écart-type)
        return np.sqrt(forecasts)

    def get_volatility_series(self):
        """
        Retourne la série complète des volatilités conditionnelles.

        Retourne:
        ---------
        np.array : volatilités conditionnelles estimées
        """
        if self.conditional_vol is None:
            raise ValueError("Le modèle doit être estimé d'abord")

        return self.conditional_vol


def ewma_volatility(returns, lambda_param=0.94):
    """
    Calcule la volatilité EWMA (Exponentially Weighted Moving Average).

    Modèle EWMA (cas particulier de GARCH avec ω=0):
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}

    RiskMetrics utilise λ = 0.94 pour les données journalières.

    Paramètres:
    -----------
    returns : array-like
        Rendements
    lambda_param : float
        Paramètre de lissage (défaut: 0.94 = RiskMetrics)

    Retourne:
    ---------
    np.array : volatilités EWMA
    """
    returns = np.array(returns)
    n = len(returns)
    variance = np.zeros(n)

    # Initialisation
    variance[0] = returns[0]**2

    # Récurrence EWMA
    for t in range(1, n):
        variance[t] = lambda_param * variance[t-1] + (1 - lambda_param) * returns[t-1]**2

    return np.sqrt(variance)


# Tests si exécuté directement
if __name__ == "__main__":
    # Générer des données avec volatilité variable (modèle GARCH simulé)
    np.random.seed(42)
    n = 1000

    # Vrais paramètres
    true_omega = 0.00001
    true_alpha = 0.10
    true_beta = 0.85

    # Simulation GARCH
    returns = np.zeros(n)
    variance = np.zeros(n)
    variance[0] = true_omega / (1 - true_alpha - true_beta)

    for t in range(1, n):
        variance[t] = true_omega + true_alpha * returns[t-1]**2 + true_beta * variance[t-1]
        returns[t] = np.sqrt(variance[t]) * np.random.standard_normal()

    print("=== Test GARCH(1,1) ===")
    print(f"\nVrais paramètres:")
    print(f"  ω = {true_omega:.6f}")
    print(f"  α = {true_alpha:.2f}")
    print(f"  β = {true_beta:.2f}")

    # Estimer le modèle
    model = GARCHModel(returns)
    result = model.fit()

    print(f"\nParamètres estimés:")
    print(f"  ω = {result['omega']:.6f}")
    print(f"  α = {result['alpha']:.4f}")
    print(f"  β = {result['beta']:.4f}")
    print(f"  Persistance (α+β) = {result['persistence']:.4f}")
    print(f"  Vol long-terme = {result['long_term_vol']*100:.2f}%")
    print(f"  Convergence: {result['converged']}")

    # Prévision
    print(f"\nPrévision de volatilité:")
    forecasts = model.forecast(horizon=5)
    for i, vol in enumerate(forecasts):
        print(f"  Jour +{i+1}: {vol*100:.4f}%")

    # Comparaison avec EWMA
    print(f"\n--- Comparaison avec EWMA (λ=0.94) ---")
    ewma_vol = ewma_volatility(returns, 0.94)
    garch_vol = model.get_volatility_series()

    print(f"Vol finale GARCH: {garch_vol[-1]*100:.4f}%")
    print(f"Vol finale EWMA:  {ewma_vol[-1]*100:.4f}%")
