"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.
"""

import numpy as np

from algorithms.algorithm import Algorithm


class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.

        Selecciona el brazo que maximiza el límite superior de confianza.

        El parámetro c controla el grado de exploración (usualmente c=1).
        Un c mayor favorece la exploración; c menor, la explotación.

        :param k: Número de brazos.
        :param c: Parámetro de exploración (debe ser >= 0).
        """
        assert c >= 0, "El parámetro c debe ser mayor o igual a 0."

        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona el brazo con mayor valor UCB1.

        :return: Índice del brazo seleccionado.
        """
        # Primer recorrido: seleccionar brazos no visitados antes de calcular UCB1.
        # Se elige aleatoriamente entre los no visitados para evitar sesgo de orden.
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited) > 0:
            return int(np.random.choice(unvisited))

        # Número total de pasos realizados
        t = int(np.sum(self.counts))

        # UCB1(a) = Q(a) + c * sqrt(ln(t) / N(a))
        ucb_values = self.values + self.c * np.sqrt(np.log(t) / self.counts)

        return int(np.argmax(ucb_values))
