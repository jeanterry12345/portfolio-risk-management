# -*- coding: utf-8 -*-
"""
Démonstration de la Plateforme de Risk Management
=================================================
Script de démonstration avec données simulées CAC40.

Ce script montre:
1. Calcul du VaR par les trois méthodes
2. CVaR / Expected Shortfall
3. Décomposition du risque
4. Estimation GARCH
5. Stress Testing
6. Backtesting
7. Génération de rapport Excel
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.var_calculator import VaRCalculator, var_portfolio
from src.cvar_calculator import calculate_cvar, var_cvar_comparison
from src.marginal_component import component_var, risk_decomposition_report
from src.garch_model import GARCHModel, ewma_volatility
from src.stress_testing import StressTester
from src.backtesting import kupiec_test, christoffersen_test, backtest_report
from src.reporting import generate_risk_report


def generate_cac40_data(n_days=504, seed=42):
    """
    Génère des données simulées pour un portefeuille CAC40.

    Simule les rendements de 5 actions françaises avec des
    corrélations réalistes.

    Paramètres:
    -----------
    n_days : int
        Nombre de jours (défaut: 504 = 2 ans)
    seed : int
        Graine aléatoire

    Retourne:
    ---------
    tuple : (asset_names, returns_matrix, weights)
    """
    np.random.seed(seed)

    # Actions CAC40 simulées
    asset_names = ['TotalEnergies', 'LVMH', 'Airbus', 'Sanofi', 'BNP Paribas']

    # Volatilités annuelles typiques
    annual_vols = np.array([0.25, 0.28, 0.32, 0.20, 0.35])
    daily_vols = annual_vols / np.sqrt(252)

    # Matrice de corrélation réaliste
    corr_matrix = np.array([
        [1.00, 0.35, 0.40, 0.25, 0.45],  # TTE
        [0.35, 1.00, 0.30, 0.20, 0.40],  # LVMH
        [0.40, 0.30, 1.00, 0.25, 0.35],  # Airbus
        [0.25, 0.20, 0.25, 1.00, 0.30],  # Sanofi
        [0.45, 0.40, 0.35, 0.30, 1.00]   # BNP
    ])

    # Matrice de covariance
    cov_matrix = np.outer(daily_vols, daily_vols) * corr_matrix

    # Générer des rendements corrélés
    L = np.linalg.cholesky(cov_matrix)
    returns_uncorr = np.random.normal(0, 1, (n_days, 5))
    returns_matrix = np.dot(returns_uncorr, L.T)

    # Ajouter quelques événements extrêmes (simulation de crises)
    # Mini-crise autour du jour 200
    returns_matrix[200:210, :] *= 2.5
    returns_matrix[205, :] -= 0.03  # Journée très négative

    # Poids du portefeuille
    weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])

    return asset_names, returns_matrix, weights


def demo_var_calculation():
    """
    Démonstration 1: Calcul du VaR.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 1: CALCUL DU VaR")
    print("=" * 60)

    # Générer les données
    asset_names, returns_matrix, weights = generate_cac40_data()

    # Rendements du portefeuille
    portfolio_returns = np.dot(returns_matrix, weights)

    print(f"\nStatistiques du portefeuille:")
    print(f"  Rendement moyen: {np.mean(portfolio_returns)*252*100:.2f}% annualisé")
    print(f"  Volatilité: {np.std(portfolio_returns)*np.sqrt(252)*100:.2f}% annualisé")
    print(f"  Min: {np.min(portfolio_returns)*100:.2f}%")
    print(f"  Max: {np.max(portfolio_returns)*100:.2f}%")

    # Calculer le VaR
    calculator = VaRCalculator(portfolio_returns, portfolio_value=1000000)

    print(f"\n=== VaR sur 1 jour ===")
    for confidence in [0.95, 0.99]:
        results = calculator.calculate_all(confidence)
        print(f"\nVaR {confidence*100:.0f}%:")
        print(f"  Historique:   {results['historical']*100:.4f}% ({results['historical']*1000000:,.0f} €)")
        print(f"  Paramétrique: {results['parametric']*100:.4f}% ({results['parametric']*1000000:,.0f} €)")
        print(f"  Monte Carlo:  {results['monte_carlo']*100:.4f}% ({results['monte_carlo']*1000000:,.0f} €)")

    return calculator


def demo_cvar():
    """
    Démonstration 2: CVaR / Expected Shortfall.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 2: CVaR / EXPECTED SHORTFALL")
    print("=" * 60)

    asset_names, returns_matrix, weights = generate_cac40_data()
    portfolio_returns = np.dot(returns_matrix, weights)

    print("\nComparaison VaR vs CVaR:")

    for confidence in [0.95, 0.99]:
        comparison = var_cvar_comparison(portfolio_returns, confidence)

        print(f"\nNiveau de confiance: {confidence*100:.0f}%")
        print(f"  VaR:  {comparison['var']*100:.4f}%")
        print(f"  CVaR: {comparison['cvar']*100:.4f}%")
        print(f"  Ratio CVaR/VaR: {comparison['ratio']:.2f}")
        print(f"  Excès CVaR-VaR: {comparison['excess']*100:.4f}%")


def demo_risk_decomposition():
    """
    Démonstration 3: Décomposition du risque.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 3: DÉCOMPOSITION DU RISQUE")
    print("=" * 60)

    asset_names, returns_matrix, weights = generate_cac40_data()

    # Rapport de décomposition
    report = risk_decomposition_report(asset_names, returns_matrix, weights, 0.95)
    print(report)


def demo_garch():
    """
    Démonstration 4: Modèle GARCH(1,1).
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 4: MODÈLE GARCH(1,1)")
    print("=" * 60)

    asset_names, returns_matrix, weights = generate_cac40_data()
    portfolio_returns = np.dot(returns_matrix, weights)

    # Estimer le modèle GARCH
    print("\nEstimation GARCH(1,1)...")
    model = GARCHModel(portfolio_returns)
    result = model.fit()

    print(f"\nParamètres estimés:")
    print(f"  ω (omega): {result['omega']:.8f}")
    print(f"  α (alpha): {result['alpha']:.4f}")
    print(f"  β (beta):  {result['beta']:.4f}")
    print(f"  Persistance (α+β): {result['persistence']:.4f}")
    print(f"  Vol long-terme: {result['long_term_vol']*np.sqrt(252)*100:.2f}% annualisé")

    # Prévision
    print(f"\nPrévision de volatilité (jours suivants):")
    forecasts = model.forecast(horizon=5)
    for i, vol in enumerate(forecasts):
        print(f"  Jour +{i+1}: {vol*100:.4f}% ({vol*np.sqrt(252)*100:.2f}% annualisé)")

    # Comparaison EWMA
    ewma_vol = ewma_volatility(portfolio_returns, lambda_param=0.94)
    garch_vol = model.get_volatility_series()

    print(f"\nDernière volatilité:")
    print(f"  GARCH: {garch_vol[-1]*100:.4f}%")
    print(f"  EWMA:  {ewma_vol[-1]*100:.4f}%")


def demo_stress_testing():
    """
    Démonstration 5: Stress Testing.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 5: STRESS TESTING")
    print("=" * 60)

    assets = ['Actions CAC40', 'Obligations État', 'Immobilier', 'Or']
    weights = [0.60, 0.25, 0.10, 0.05]
    portfolio_value = 1000000

    tester = StressTester(assets, weights, portfolio_value)

    # Résumé de tous les scénarios
    print("\nRésumé des scénarios de stress:")
    summary = tester.run_all_scenarios()
    print(summary.to_string(index=False))

    # Détail d'un scénario
    print(tester.generate_report('2008_crisis'))


def demo_backtesting():
    """
    Démonstration 6: Backtesting du VaR.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 6: BACKTESTING DU VaR")
    print("=" * 60)

    # Générer les données
    asset_names, returns_matrix, weights = generate_cac40_data(n_days=504)
    portfolio_returns = np.dot(returns_matrix, weights)

    # Calculer le VaR rolling
    window = 60
    n = len(portfolio_returns)
    var_estimates = np.zeros(n)

    for t in range(window, n):
        historical_returns = portfolio_returns[t-window:t]
        std = np.std(historical_returns)
        var_estimates[t] = 1.645 * std  # VaR 95%

    # Backtesting sur la partie avec VaR calculé
    returns_test = portfolio_returns[window:]
    var_test = var_estimates[window:]

    # Afficher le rapport
    print(backtest_report(returns_test, var_test, 0.95, 'Rolling 60 jours'))


def demo_reporting():
    """
    Démonstration 7: Génération de rapport Excel.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 7: GÉNÉRATION DE RAPPORT EXCEL")
    print("=" * 60)

    # Données du portefeuille
    asset_names, returns_matrix, weights = generate_cac40_data()
    portfolio_returns = np.dot(returns_matrix, weights)
    portfolio_value = 1000000

    calculator = VaRCalculator(portfolio_returns)

    # Préparer les données pour le rapport
    portfolio_data = {
        'assets': asset_names,
        'weights': weights.tolist(),
        'value': portfolio_value
    }

    var_results = {
        'var_95': calculator.var_parametric(0.95),
        'var_99': calculator.var_parametric(0.99),
        'cvar_95': calculate_cvar(portfolio_returns, 0.95),
        'var_95_hist': calculator.var_historical(0.95),
        'var_95_param': calculator.var_parametric(0.95),
        'var_95_mc': calculator.var_monte_carlo(0.95),
        'var_99_hist': calculator.var_historical(0.99),
        'var_99_param': calculator.var_parametric(0.99),
        'var_99_mc': calculator.var_monte_carlo(0.99)
    }

    # Component VaR
    comp_var = component_var(returns_matrix, weights, 0.95)
    var_results['marginal_var'] = comp_var['marginal_var'].tolist()
    var_results['component_var'] = comp_var['component_var'].tolist()
    var_results['contribution'] = comp_var['percent_contribution'].tolist()

    # Stress testing
    tester = StressTester(asset_names, weights.tolist(), portfolio_value)
    stress_results = []
    for scenario in ['2008_crisis', 'covid_2020', 'rate_shock_up']:
        result = tester.run_scenario(scenario)
        stress_results.append({
            'name': result['scenario_name'],
            'return': result['portfolio_return'],
            'loss': result['loss'],
            'final_value': result['surviving_value']
        })

    # Backtesting
    window = 60
    n = len(portfolio_returns)
    var_estimates = np.zeros(n)
    for t in range(window, n):
        std = np.std(portfolio_returns[t-window:t])
        var_estimates[t] = 1.645 * std

    returns_test = portfolio_returns[window:]
    var_test = var_estimates[window:]
    kupiec = kupiec_test(returns_test, var_test, 0.95)
    chris = christoffersen_test(returns_test, var_test, 0.95)

    backtest_results = {
        'period': 'Simulation 2 ans',
        'n_obs': kupiec['n_observations'],
        'n_exceptions': kupiec['n_exceptions'],
        'expected_exceptions': kupiec['expected_exceptions'],
        'exception_rate': kupiec['exception_rate_observed'],
        'kupiec_passed': kupiec['passed'],
        'kupiec_pvalue': kupiec['p_value'],
        'chris_passed': chris['passed_combined'],
        'chris_pvalue': chris['p_value_combined'],
        'conclusion': chris['conclusion']
    }

    # Générer le rapport
    filename = 'demo_rapport_risque.xlsx'
    generate_risk_report(filename, portfolio_data, var_results, stress_results, backtest_results)
    print(f"\nRapport Excel généré: {filename}")


def main():
    """
    Exécute toutes les démonstrations.
    """
    print("\n" + "#" * 70)
    print("#" + " " * 15 + "PLATEFORME DE GESTION DU RISQUE" + " " * 20 + "#")
    print("#" + " " * 15 + "Démonstration des Fonctionnalités" + " " * 17 + "#")
    print("#" * 70)

    demo_var_calculation()
    demo_cvar()
    demo_risk_decomposition()
    demo_garch()
    demo_stress_testing()
    demo_backtesting()
    demo_reporting()

    print("\n" + "=" * 60)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 60)
    print("\nFichiers générés:")
    print("  - demo_rapport_risque.xlsx")


if __name__ == "__main__":
    main()
