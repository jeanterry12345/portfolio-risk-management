# -*- coding: utf-8 -*-
"""
Portfolio Risk Management
=========================
Plateforme de gestion du risque de portefeuille.

Modules:
    - var_calculator: Calcul du VaR (historique, paramétrique, Monte Carlo)
    - cvar_calculator: CVaR / Expected Shortfall
    - marginal_component: Marginal VaR et Component VaR
    - garch_model: Estimation GARCH(1,1)
    - stress_testing: Scénarios de stress
    - backtesting: Tests de Kupiec et Christoffersen
    - reporting: Export Excel
"""

from .var_calculator import VaRCalculator
from .cvar_calculator import calculate_cvar, calculate_expected_shortfall
from .marginal_component import marginal_var, component_var
from .garch_model import GARCHModel
from .stress_testing import StressTester
from .backtesting import kupiec_test, christoffersen_test
from .reporting import generate_risk_report

__version__ = "1.0.0"
__author__ = "M2 Finance Quantitative - Sorbonne"
