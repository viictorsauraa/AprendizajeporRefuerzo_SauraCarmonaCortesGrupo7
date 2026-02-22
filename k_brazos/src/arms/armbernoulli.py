"""
Module: arms/armbernoulli.py
Description: Contains the implementation of the ArmBernoulli class for the Bernoulli distribution arm.
"""


import numpy as np

from arms.armbinomial import ArmBinomial


class ArmBernoulli(ArmBinomial):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución Bernoulli.
        La distribución Bernoulli es un caso particular de la binomial con n=1.

        :param p: Probabilidad de éxito (recompensa = 1).
        """
        super().__init__(n=1, p=p)

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernoulli(p={self.p:.3f})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Valor mínimo de la probabilidad.
        :param p_max: Valor máximo de la probabilidad.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 < p_min < p_max <= 1, "Los valores de p_min y p_max deben estar en (0, 1] con p_min < p_max."

        # Generar k valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 3)
            p_values.add(p)

        p_values = list(p_values)

        arms = [ArmBernoulli(p) for p in p_values]

        return arms
