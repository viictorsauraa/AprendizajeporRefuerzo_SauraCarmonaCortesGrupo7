"""
Module: arms/armbernoulli.py
Description: Contains the implementation of the ArmBernoulli class for the Bernoulli distribution arm.
"""


import numpy as np

from arms import Arm


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución Bernoulli.
        La distribución Bernoulli es un caso particular de la binomial con n=1.

        :param p: Probabilidad de éxito (recompensa = 1).
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0, 1]."

        self.p = p
        self.n = 1 # Definimos n=1 como constante para la distribución Bernoulli

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Bernoulli.
        Retorna 1 (éxito) con probabilidad p, y 0 (fracaso) con probabilidad 1-p.

        :return: Recompensa obtenida del brazo (0 o 1).
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.
        Para una distribución Bernoulli, el valor esperado es μ = p.

        :return: Valor esperado de la distribución.
        """
        return self.p

    def get_variance(self) -> float:
        """
        Devuelve la varianza de la distribución Bernoulli.
        Para una distribución Bernoulli, la varianza es σ² = p(1-p).

        :return: Varianza de la distribución.
        """
        return self.p * (1 - self.p)

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