"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the binomial distribution arm.
"""


import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de ensayos (acciones).
        :param p: Probabilidad de éxito en cada ensayo.
        """
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0, 1]."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo (número de éxitos en n ensayos).
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.
        Para una distribución binomial B(n, p), el valor esperado es μ = np.

        :return: Valor esperado de la distribución.
        """
        return self.n * self.p
    
    def get_variance(self) -> float:
        """
        Devuelve la varianza de la distribución binomial.
        Para una distribución binomial B(n, p), la varianza es σ² = np(1-p).

        :return: Varianza de la distribución.
        """
        return self.n * self.p * (1 - self.p)
    
    def can_approximate_normal(self) -> bool:
        """
        Verifica si la distribución binomial puede aproximarse a una normal.
        Condiciones: np ≥ 5 y n(1-p) ≥ 5.

        :return: True si se puede aproximar a una normal, False en caso contrario.
        """
        return (self.n * self.p >= 5) and (self.n * (1 - self.p) >= 5)

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        approx_info = " (aproximable a normal)" if self.can_approximate_normal() else ""
        return f"ArmBinomial(n={self.n}, p={self.p:.3f}, μ={self.get_expected_value():.3f}){approx_info}"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param n: Número de ensayos para cada brazo.
        :param p_min: Valor mínimo de la probabilidad.
        :param p_max: Valor máximo de la probabilidad.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 < p_min < p_max <= 1, "Los valores de p_min y p_max deben estar en (0, 1] con p_min < p_max."
        assert n > 0, "El número de ensayos n debe ser mayor que 0."

        # Generar k valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 3)
            p_values.add(p)

        p_values = list(p_values)

        arms = [ArmBinomial(n, p) for p in p_values]

        return arms


