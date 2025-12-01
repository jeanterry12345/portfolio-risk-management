# -*- coding: utf-8 -*-
"""
Tests Unitaires pour la Plateforme de Risk Management
=====================================================
Tests basiques pour valider les calculs de VaR et autres mesures de risque.
"""

import sys
import os
import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.var_calculator import VaRCalculator, var_portfolio
from src.cvar_calculator import calculate_cvar, var_cvar_comparison
from src.backtesting import kupiec_test, count_exceptions


def test_var_positive():
    """
    Test que le VaR est toujours positif (c'est une perte).
    """
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 252)

    calculator = VaRCalculator(returns)

    var_hist = calculator.var_historical(0.95)
    var_param = calculator.var_parametric(0.95)
    var_mc = calculator.var_monte_carlo(0.95, seed=42)

    assert var_hist > 0, f"VaR historique négatif: {var_hist}"
    assert var_param > 0, f"VaR paramétrique négatif: {var_param}"
    assert var_mc > 0, f"VaR Monte Carlo négatif: {var_mc}"

    print(f"✓ Test VaR positif: Hist={var_hist:.4f}, Param={var_param:.4f}, MC={var_mc:.4f}")


def test_var_99_greater_than_95():
    """
    Test que le VaR 99% est supérieur au VaR 95%.
    """
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 252)

    calculator = VaRCalculator(returns)

    var_95 = calculator.var_parametric(0.95)
    var_99 = calculator.var_parametric(0.99)

    assert var_99 > var_95, f"VaR 99% ({var_99}) <= VaR 95% ({var_95})"

    print(f"✓ Test VaR 99% > VaR 95%: {var_99:.4f} > {var_95:.4f}")


def test_cvar_greater_than_var():
    """
    Test que le CVaR est toujours >= VaR.
    """
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 252)

    comparison = var_cvar_comparison(returns, 0.95)

    assert comparison['cvar'] >= comparison['var'], \
        f"CVaR ({comparison['cvar']}) < VaR ({comparison['var']})"

    print(f"✓ Test CVaR >= VaR: {comparison['cvar']:.4f} >= {comparison['var']:.4f}")


def test_var_scaling():
    """
    Test de l'ajustement en racine du temps.

    VaR(10 jours) ≈ VaR(1 jour) * √10
    """
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 252)

    calculator = VaRCalculator(returns)

    var_1d = calculator.var_parametric(0.95, horizon=1)
    var_10d = calculator.var_parametric(0.95, horizon=10)

    # Ratio devrait être proche de √10 ≈ 3.16
    ratio = var_10d / var_1d
    expected_ratio = np.sqrt(10)

    # Tolérance de 10% (car les rendements ne sont pas exactement normaux)
    assert abs(ratio - expected_ratio) / expected_ratio < 0.1, \
        f"Ratio {ratio:.2f} trop différent de √10={expected_ratio:.2f}"

    print(f"✓ Test scaling temporel: ratio={ratio:.2f}, attendu≈{expected_ratio:.2f}")


def test_exception_count():
    """
    Test du comptage des exceptions.
    """
    # Créer des données avec des exceptions connues
    returns = np.array([0.01, 0.02, -0.05, 0.01, -0.06, 0.02])
    var_estimates = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03])

    # Exceptions: quand -return > VaR
    # -(-0.05) = 0.05 > 0.03 ✓
    # -(-0.06) = 0.06 > 0.03 ✓
    # Donc 2 exceptions attendues

    exc = count_exceptions(returns, var_estimates)

    assert exc['n_exceptions'] == 2, \
        f"Nombre d'exceptions incorrect: {exc['n_exceptions']} (attendu: 2)"

    print(f"✓ Test comptage exceptions: {exc['n_exceptions']} exceptions détectées")


def test_kupiec_perfect_model():
    """
    Test de Kupiec avec un modèle "parfait".

    Si le taux d'exceptions est exactement 5% (pour VaR 95%),
    le test devrait passer.
    """
    np.random.seed(42)
    n = 1000

    # Créer des rendements où exactement 5% sont des exceptions
    returns = np.zeros(n)
    var_estimates = np.ones(n) * 0.02

    # 5% d'exceptions (50 sur 1000)
    n_exceptions = int(n * 0.05)
    returns[:n_exceptions] = -0.03  # Ces rendements dépassent le VaR

    result = kupiec_test(returns, var_estimates, 0.95)

    # Le test devrait passer
    assert result['passed'], \
        f"Kupiec devrait passer avec taux d'exceptions parfait"

    print(f"✓ Test Kupiec modèle parfait: passé (p-value={result['p_value']:.4f})")


def test_var_portfolio_diversification():
    """
    Test que la diversification réduit le risque.

    Le VaR d'un portefeuille diversifié devrait être <= moyenne pondérée des VaR individuels.
    """
    np.random.seed(42)
    n_days = 252

    # Deux actifs peu corrélés
    returns_1 = np.random.normal(0, 0.02, n_days)
    returns_2 = np.random.normal(0, 0.02, n_days)

    # Corrélation artificielle faible
    returns_2 = 0.3 * returns_1 + 0.95 * returns_2

    returns_matrix = np.column_stack([returns_1, returns_2])
    weights = np.array([0.5, 0.5])

    # VaR du portefeuille
    var_portfolio_val = var_portfolio(returns_matrix, weights, 0.95, 'parametric')

    # VaR individuels
    var_1 = VaRCalculator(returns_1).var_parametric(0.95)
    var_2 = VaRCalculator(returns_2).var_parametric(0.95)

    # Moyenne pondérée des VaR
    var_avg = 0.5 * var_1 + 0.5 * var_2

    # Le VaR du portefeuille devrait être inférieur grâce à la diversification
    assert var_portfolio_val <= var_avg * 1.05, \
        f"VaR portefeuille ({var_portfolio_val:.4f}) > moyenne ({var_avg:.4f})"

    print(f"✓ Test diversification: VaR portfolio={var_portfolio_val:.4f} <= avg={var_avg:.4f}")


def test_var_methods_consistency():
    """
    Test que les trois méthodes de VaR donnent des résultats proches.
    """
    np.random.seed(42)
    # Beaucoup de données pour la convergence
    returns = np.random.normal(0, 0.02, 2000)

    calculator = VaRCalculator(returns)

    var_hist = calculator.var_historical(0.95)
    var_param = calculator.var_parametric(0.95)
    var_mc = calculator.var_monte_carlo(0.95, n_simulations=50000, seed=42)

    # Les trois méthodes devraient être proches (dans 20% l'une de l'autre)
    mean_var = (var_hist + var_param + var_mc) / 3

    assert abs(var_hist - mean_var) / mean_var < 0.2, "VaR historique trop différent"
    assert abs(var_param - mean_var) / mean_var < 0.2, "VaR paramétrique trop différent"
    assert abs(var_mc - mean_var) / mean_var < 0.2, "VaR Monte Carlo trop différent"

    print(f"✓ Test cohérence méthodes: Hist={var_hist:.4f}, Param={var_param:.4f}, MC={var_mc:.4f}")


def run_all_tests():
    """
    Exécute tous les tests unitaires.
    """
    print("=" * 50)
    print("TESTS UNITAIRES - Risk Management")
    print("=" * 50)
    print()

    tests = [
        test_var_positive,
        test_var_99_greater_than_95,
        test_cvar_greater_than_var,
        test_var_scaling,
        test_exception_count,
        test_kupiec_perfect_model,
        test_var_portfolio_diversification,
        test_var_methods_consistency
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} ÉCHEC: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERREUR: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Résultats: {passed} réussis, {failed} échoués sur {len(tests)}")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
