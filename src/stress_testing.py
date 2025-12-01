# -*- coding: utf-8 -*-
"""
Stress Testing
==============
Analyse de scénarios de stress pour les portefeuilles.

Référence: Hull, Chapitre 22

Le stress testing évalue l'impact de scénarios extrêmes
(mais plausibles) sur un portefeuille.

Scénarios implémentés:
    - Crise financière 2008 (Lehman Brothers)
    - Crise COVID-19 (mars 2020)
    - Scénarios hypothétiques personnalisés
"""

import numpy as np
import pandas as pd


class StressTester:
    """
    Classe pour effectuer des tests de stress sur un portefeuille.

    Attributs:
    ----------
    asset_names : list
        Noms des actifs
    weights : np.array
        Poids du portefeuille
    portfolio_value : float
        Valeur totale du portefeuille

    Exemple:
    --------
    >>> tester = StressTester(['Actions', 'Obligations'], [0.6, 0.4], 1000000)
    >>> results = tester.run_scenario('2008_crisis')
    """

    def __init__(self, asset_names, weights, portfolio_value=1000000):
        """
        Initialise le stress tester.

        Paramètres:
        -----------
        asset_names : list
            Noms des actifs dans le portefeuille
        weights : list ou np.array
            Poids de chaque actif
        portfolio_value : float
            Valeur du portefeuille en €
        """
        self.asset_names = asset_names
        self.weights = np.array(weights)
        self.portfolio_value = portfolio_value
        self.n_assets = len(asset_names)

        # Définir les scénarios de stress prédéfinis
        self._define_scenarios()

    def _define_scenarios(self):
        """
        Définit les scénarios de stress historiques.

        Les chocs sont en pourcentage de variation.
        Ex: -0.20 = baisse de 20%
        """
        # Scénario 1: Crise 2008 (septembre-octobre)
        # Chocs sur différentes classes d'actifs
        self.scenarios = {
            '2008_crisis': {
                'name': 'Crise financière 2008 (Lehman Brothers)',
                'description': 'Effondrement de Lehman Brothers, crise des subprimes',
                'date_reference': 'Septembre-Octobre 2008',
                'shocks': {
                    'equity': -0.35,        # Actions: -35%
                    'equity_financial': -0.50,  # Actions financières: -50%
                    'government_bonds': 0.05,   # Obligations d'État: +5% (flight to quality)
                    'corporate_bonds': -0.15,   # Obligations corporate: -15%
                    'commodities': -0.30,       # Matières premières: -30%
                    'real_estate': -0.25,       # Immobilier: -25%
                    'default': -0.30            # Par défaut
                }
            },

            'covid_2020': {
                'name': 'Crise COVID-19',
                'description': 'Pandémie mondiale, confinements',
                'date_reference': 'Mars 2020',
                'shocks': {
                    'equity': -0.30,            # Actions: -30%
                    'equity_tech': -0.15,       # Tech: -15% (moins touché)
                    'equity_travel': -0.60,     # Voyage/Transport: -60%
                    'government_bonds': 0.03,   # Obligations d'État: +3%
                    'corporate_bonds': -0.10,   # Obligations corporate: -10%
                    'commodities': -0.40,       # Pétrole notamment: -40%
                    'gold': 0.08,               # Or: +8% (valeur refuge)
                    'default': -0.25
                }
            },

            'rate_shock_up': {
                'name': 'Choc de taux (+200bp)',
                'description': 'Hausse brutale des taux d\'intérêt de 200 points de base',
                'date_reference': 'Scénario hypothétique',
                'shocks': {
                    'equity': -0.10,            # Actions: -10%
                    'equity_growth': -0.20,     # Growth stocks: -20%
                    'equity_value': -0.05,      # Value stocks: -5%
                    'government_bonds': -0.15,  # Obligations d'État: -15%
                    'corporate_bonds': -0.12,   # Obligations corporate: -12%
                    'real_estate': -0.15,       # Immobilier: -15%
                    'default': -0.10
                }
            },

            'geopolitical': {
                'name': 'Crise géopolitique majeure',
                'description': 'Conflit majeur, perturbation des échanges',
                'date_reference': 'Scénario hypothétique',
                'shocks': {
                    'equity': -0.25,
                    'equity_emerging': -0.40,   # Marchés émergents: -40%
                    'government_bonds': 0.02,
                    'commodities': 0.30,        # Hausse des commodités: +30%
                    'gold': 0.15,               # Or: +15%
                    'default': -0.20
                }
            }
        }

    def _get_shock_for_asset(self, scenario_name, asset_name):
        """
        Récupère le choc approprié pour un actif donné.

        Paramètres:
        -----------
        scenario_name : str
            Nom du scénario
        asset_name : str
            Nom de l'actif

        Retourne:
        ---------
        float : choc en pourcentage
        """
        scenario = self.scenarios[scenario_name]
        shocks = scenario['shocks']

        # Chercher un choc spécifique pour l'actif
        asset_lower = asset_name.lower()

        for shock_type, shock_value in shocks.items():
            if shock_type.lower() in asset_lower or asset_lower in shock_type.lower():
                return shock_value

        # Sinon, utiliser le choc par défaut
        return shocks.get('default', -0.20)

    def run_scenario(self, scenario_name, custom_shocks=None):
        """
        Exécute un scénario de stress sur le portefeuille.

        Paramètres:
        -----------
        scenario_name : str
            Nom du scénario prédéfini ('2008_crisis', 'covid_2020', etc.)
            ou 'custom' pour utiliser des chocs personnalisés
        custom_shocks : dict ou list
            Chocs personnalisés par actif (si scenario_name='custom')

        Retourne:
        ---------
        dict : résultats du stress test
        """
        if scenario_name == 'custom' and custom_shocks is not None:
            if isinstance(custom_shocks, dict):
                shocks = [custom_shocks.get(name, 0) for name in self.asset_names]
            else:
                shocks = custom_shocks
            scenario_info = {
                'name': 'Scénario personnalisé',
                'description': 'Chocs définis par l\'utilisateur',
                'date_reference': 'N/A'
            }
        else:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Scénario inconnu: {scenario_name}")

            scenario_info = self.scenarios[scenario_name]
            shocks = [self._get_shock_for_asset(scenario_name, name)
                      for name in self.asset_names]

        shocks = np.array(shocks)

        # Calcul de l'impact
        # Rendement du portefeuille sous stress
        portfolio_return = np.dot(self.weights, shocks)

        # Perte en valeur absolue
        loss = -portfolio_return * self.portfolio_value

        # Impact par actif
        asset_contributions = self.weights * shocks * self.portfolio_value

        return {
            'scenario_name': scenario_info['name'],
            'description': scenario_info['description'],
            'date_reference': scenario_info.get('date_reference', 'N/A'),
            'shocks': shocks,
            'portfolio_return': portfolio_return,
            'loss': loss,
            'loss_percent': -portfolio_return * 100,
            'asset_contributions': asset_contributions,
            'surviving_value': self.portfolio_value + portfolio_return * self.portfolio_value
        }

    def run_all_scenarios(self):
        """
        Exécute tous les scénarios de stress prédéfinis.

        Retourne:
        ---------
        pd.DataFrame : résumé de tous les scénarios
        """
        results = []

        for scenario_name in self.scenarios.keys():
            result = self.run_scenario(scenario_name)
            results.append({
                'Scénario': result['scenario_name'],
                'Rendement (%)': result['portfolio_return'] * 100,
                'Perte (€)': result['loss'],
                'Valeur finale (€)': result['surviving_value']
            })

        return pd.DataFrame(results)

    def generate_report(self, scenario_name):
        """
        Génère un rapport détaillé pour un scénario.

        Paramètres:
        -----------
        scenario_name : str
            Nom du scénario

        Retourne:
        ---------
        str : rapport formaté
        """
        result = self.run_scenario(scenario_name)

        lines = [
            f"\n{'='*70}",
            f"STRESS TEST - {result['scenario_name'].upper()}",
            f"{'='*70}",
            f"\nDescription: {result['description']}",
            f"Référence: {result['date_reference']}",
            f"\nValeur initiale du portefeuille: {self.portfolio_value:,.0f} €",
            f"\n{'Actif':<20} {'Poids':<10} {'Choc':<12} {'Contribution':<15}",
            f"{'-'*60}"
        ]

        for i, name in enumerate(self.asset_names):
            lines.append(
                f"{name:<20} {self.weights[i]*100:>8.2f}% "
                f"{result['shocks'][i]*100:>10.1f}% "
                f"{result['asset_contributions'][i]:>13,.0f} €"
            )

        lines.extend([
            f"{'-'*60}",
            f"\nRÉSULTATS:",
            f"  Rendement du portefeuille: {result['portfolio_return']*100:+.2f}%",
            f"  Perte totale: {result['loss']:,.0f} €",
            f"  Valeur finale: {result['surviving_value']:,.0f} €",
            f"{'='*70}"
        ])

        return '\n'.join(lines)


def sensitivity_analysis(portfolio_value, weights, asset_names, shock_range=(-0.3, 0.1)):
    """
    Analyse de sensibilité: impact de différents niveaux de choc.

    Paramètres:
    -----------
    portfolio_value : float
        Valeur du portefeuille
    weights : np.array
        Poids
    asset_names : list
        Noms des actifs
    shock_range : tuple
        Plage de chocs à tester

    Retourne:
    ---------
    pd.DataFrame : tableau de sensibilité
    """
    shock_levels = np.linspace(shock_range[0], shock_range[1], 9)
    results = []

    for shock in shock_levels:
        # Appliquer le même choc à tous les actifs
        portfolio_return = np.dot(weights, np.ones(len(weights)) * shock)
        loss = -portfolio_return * portfolio_value

        results.append({
            'Choc (%)': shock * 100,
            'Rendement (%)': portfolio_return * 100,
            'Perte (€)': loss,
            'Valeur finale (€)': portfolio_value * (1 + portfolio_return)
        })

    return pd.DataFrame(results)


# Tests si exécuté directement
if __name__ == "__main__":
    # Portefeuille exemple CAC40
    assets = ['Actions CAC40', 'Obligations État', 'Immobilier', 'Or']
    weights = [0.50, 0.30, 0.15, 0.05]
    portfolio_value = 1000000  # 1 million €

    print("=== Test Stress Testing ===")
    print(f"\nPortefeuille initial: {portfolio_value:,.0f} €")
    print(f"Allocation: {dict(zip(assets, [f'{w*100:.0f}%' for w in weights]))}")

    # Créer le stress tester
    tester = StressTester(assets, weights, portfolio_value)

    # Exécuter tous les scénarios
    print("\n" + "="*70)
    print("RÉSUMÉ DE TOUS LES SCÉNARIOS")
    print("="*70)
    summary = tester.run_all_scenarios()
    print(summary.to_string(index=False))

    # Rapport détaillé pour 2008
    print(tester.generate_report('2008_crisis'))

    # Rapport détaillé pour COVID
    print(tester.generate_report('covid_2020'))

    # Analyse de sensibilité
    print("\n" + "="*70)
    print("ANALYSE DE SENSIBILITÉ")
    print("="*70)
    sensitivity = sensitivity_analysis(portfolio_value, np.array(weights), assets)
    print(sensitivity.to_string(index=False))
