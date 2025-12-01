# -*- coding: utf-8 -*-
"""
Reporting Excel
===============
Export automatisé des résultats de risque vers Excel.

Fonctionnalités:
    - Rapport VaR/CVaR
    - Décomposition du risque
    - Résultats de stress testing
    - Résultats de backtesting
"""

import numpy as np
import pandas as pd
from datetime import datetime


def generate_risk_report(filename, portfolio_data, var_results, stress_results=None,
                          backtest_results=None):
    """
    Génère un rapport Excel complet de gestion du risque.

    Paramètres:
    -----------
    filename : str
        Nom du fichier Excel de sortie
    portfolio_data : dict
        Données du portefeuille (assets, weights, value)
    var_results : dict
        Résultats des calculs de VaR
    stress_results : dict, optional
        Résultats des stress tests
    backtest_results : dict, optional
        Résultats du backtesting

    Retourne:
    ---------
    str : chemin du fichier créé

    Exemple:
    --------
    >>> generate_risk_report('risk_report.xlsx', portfolio, var, stress, backtest)
    """
    # Créer un writer Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:

        # === Onglet 1: Résumé ===
        summary_data = {
            'Métrique': [
                'Date du rapport',
                'Valeur du portefeuille',
                'VaR 95% (1 jour)',
                'VaR 99% (1 jour)',
                'CVaR 95%',
                'Nombre d\'actifs'
            ],
            'Valeur': [
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                f"{portfolio_data.get('value', 0):,.0f} €",
                f"{var_results.get('var_95', 0)*100:.2f}%",
                f"{var_results.get('var_99', 0)*100:.2f}%",
                f"{var_results.get('cvar_95', 0)*100:.2f}%",
                len(portfolio_data.get('assets', []))
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Résumé', index=False)

        # === Onglet 2: Composition du portefeuille ===
        assets = portfolio_data.get('assets', [])
        weights = portfolio_data.get('weights', [])
        value = portfolio_data.get('value', 0)

        portfolio_composition = {
            'Actif': assets,
            'Poids (%)': [w * 100 for w in weights],
            'Valeur (€)': [w * value for w in weights]
        }
        df_portfolio = pd.DataFrame(portfolio_composition)
        df_portfolio.to_excel(writer, sheet_name='Portefeuille', index=False)

        # === Onglet 3: VaR détaillé ===
        var_detailed = {
            'Méthode': ['Historique', 'Paramétrique', 'Monte Carlo'],
            'VaR 95% (%)': [
                var_results.get('var_95_hist', 0) * 100,
                var_results.get('var_95_param', 0) * 100,
                var_results.get('var_95_mc', 0) * 100
            ],
            'VaR 95% (€)': [
                var_results.get('var_95_hist', 0) * value,
                var_results.get('var_95_param', 0) * value,
                var_results.get('var_95_mc', 0) * value
            ],
            'VaR 99% (%)': [
                var_results.get('var_99_hist', 0) * 100,
                var_results.get('var_99_param', 0) * 100,
                var_results.get('var_99_mc', 0) * 100
            ],
            'VaR 99% (€)': [
                var_results.get('var_99_hist', 0) * value,
                var_results.get('var_99_param', 0) * value,
                var_results.get('var_99_mc', 0) * value
            ]
        }
        df_var = pd.DataFrame(var_detailed)
        df_var.to_excel(writer, sheet_name='VaR Détaillé', index=False)

        # === Onglet 4: Décomposition du risque ===
        if 'component_var' in var_results:
            decomposition = {
                'Actif': assets,
                'Marginal VaR (%)': [v * 100 for v in var_results.get('marginal_var', [])],
                'Component VaR (%)': [v * 100 for v in var_results.get('component_var', [])],
                'Contribution (%)': var_results.get('contribution', [])
            }
            df_decomp = pd.DataFrame(decomposition)
            df_decomp.to_excel(writer, sheet_name='Décomposition', index=False)

        # === Onglet 5: Stress Testing ===
        if stress_results is not None:
            stress_data = []
            for scenario in stress_results:
                stress_data.append({
                    'Scénario': scenario['name'],
                    'Rendement (%)': scenario['return'] * 100,
                    'Perte (€)': scenario['loss'],
                    'Valeur finale (€)': scenario['final_value']
                })
            df_stress = pd.DataFrame(stress_data)
            df_stress.to_excel(writer, sheet_name='Stress Testing', index=False)

        # === Onglet 6: Backtesting ===
        if backtest_results is not None:
            backtest_data = {
                'Métrique': [
                    'Période de test',
                    'Nombre d\'observations',
                    'Exceptions observées',
                    'Exceptions attendues',
                    'Taux d\'exceptions',
                    'Test de Kupiec',
                    'P-value Kupiec',
                    'Test de Christoffersen',
                    'P-value Christoffersen',
                    'Conclusion'
                ],
                'Valeur': [
                    backtest_results.get('period', 'N/A'),
                    backtest_results.get('n_obs', 0),
                    backtest_results.get('n_exceptions', 0),
                    backtest_results.get('expected_exceptions', 0),
                    f"{backtest_results.get('exception_rate', 0)*100:.2f}%",
                    'Passé' if backtest_results.get('kupiec_passed', False) else 'Échoué',
                    f"{backtest_results.get('kupiec_pvalue', 0):.4f}",
                    'Passé' if backtest_results.get('chris_passed', False) else 'Échoué',
                    f"{backtest_results.get('chris_pvalue', 0):.4f}",
                    backtest_results.get('conclusion', 'N/A')
                ]
            }
            df_backtest = pd.DataFrame(backtest_data)
            df_backtest.to_excel(writer, sheet_name='Backtesting', index=False)

    return filename


def create_sample_report(filename='rapport_risque.xlsx'):
    """
    Crée un rapport Excel d'exemple avec des données fictives.

    Utile pour tester le format du rapport.

    Paramètres:
    -----------
    filename : str
        Nom du fichier de sortie

    Retourne:
    ---------
    str : chemin du fichier créé
    """
    # Données du portefeuille
    portfolio_data = {
        'assets': ['TotalEnergies', 'LVMH', 'Airbus', 'Sanofi', 'BNP Paribas'],
        'weights': [0.25, 0.20, 0.20, 0.20, 0.15],
        'value': 1000000
    }

    # Résultats VaR
    var_results = {
        'var_95': 0.0245,
        'var_99': 0.0358,
        'cvar_95': 0.0312,
        'var_95_hist': 0.0238,
        'var_95_param': 0.0245,
        'var_95_mc': 0.0252,
        'var_99_hist': 0.0345,
        'var_99_param': 0.0358,
        'var_99_mc': 0.0365,
        'marginal_var': [0.028, 0.032, 0.035, 0.022, 0.038],
        'component_var': [0.007, 0.0064, 0.007, 0.0044, 0.0057],
        'contribution': [28.6, 26.1, 28.6, 18.0, 23.3]
    }

    # Stress tests
    stress_results = [
        {'name': 'Crise 2008', 'return': -0.32, 'loss': 320000, 'final_value': 680000},
        {'name': 'COVID-19', 'return': -0.25, 'loss': 250000, 'final_value': 750000},
        {'name': 'Choc taux +200bp', 'return': -0.12, 'loss': 120000, 'final_value': 880000}
    ]

    # Backtesting
    backtest_results = {
        'period': '2022-01-01 à 2023-12-31',
        'n_obs': 504,
        'n_exceptions': 28,
        'expected_exceptions': 25.2,
        'exception_rate': 0.0556,
        'kupiec_passed': True,
        'kupiec_pvalue': 0.4521,
        'chris_passed': True,
        'chris_pvalue': 0.3845,
        'conclusion': 'VaR validé'
    }

    return generate_risk_report(filename, portfolio_data, var_results,
                                stress_results, backtest_results)


def export_timeseries_to_excel(filename, dates, returns, var_series, vol_series=None):
    """
    Exporte les séries temporelles vers Excel.

    Paramètres:
    -----------
    filename : str
        Nom du fichier
    dates : array-like
        Dates
    returns : array-like
        Rendements
    var_series : array-like
        Série de VaR
    vol_series : array-like, optional
        Série de volatilité

    Retourne:
    ---------
    str : chemin du fichier créé
    """
    data = {
        'Date': dates,
        'Rendement': returns,
        'VaR': var_series
    }

    if vol_series is not None:
        data['Volatilité'] = vol_series

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, sheet_name='Données')

    return filename


# Tests si exécuté directement
if __name__ == "__main__":
    print("=== Test Reporting Excel ===")

    # Créer un rapport d'exemple
    filename = create_sample_report('test_rapport_risque.xlsx')
    print(f"\nRapport créé: {filename}")
    print("\nOnglets du rapport:")
    print("  1. Résumé")
    print("  2. Portefeuille")
    print("  3. VaR Détaillé")
    print("  4. Décomposition")
    print("  5. Stress Testing")
    print("  6. Backtesting")
