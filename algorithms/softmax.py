"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo Softmax para el problema de los k-brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm


class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo Softmax (Gibbs).

        La temperatura tau controla el balance exploración-explotación:
        - Si tau alto = probabilidades más uniformes (más exploración)
        - Si tau bajo = probabilidades concentradas en el mejor brazo (más explotación)
        - Si tau es 0 = equivalente a greedy
        - Si tau es ∞ = selección uniforme aleatoria

        :param k: Número de brazos.
        :param tau: Temperatura (debe ser > 0).
        """
        assert tau > 0, "La temperatura tau debe ser mayor que 0."

        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo muestreando de la distribución de Gibbs (softmax).

        :return: Índice del brazo seleccionado.
        """
        # Restamos el máximo para evita overflow en el cálculo de exponenciales
        shifted = self.values - np.max(self.values)
        exp_values = np.exp(shifted / self.tau)
        probabilities = exp_values / np.sum(exp_values)

        return np.random.choice(self.k, p=probabilities)
