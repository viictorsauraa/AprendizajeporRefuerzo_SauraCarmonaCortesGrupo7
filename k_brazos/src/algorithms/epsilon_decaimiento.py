"""
Module: algorithms/epsilon_decaimiento.py
Description: Implementación del algoritmo epsilon-greedy con decaimiento para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class EpsilonDecaimiento(Algorithm):

    def __init__(self, k: int, epsilon_initial: float = 1.0, lambda_: float = 0.01, epsilon_min: float = 0.0):
        """
        Inicializa el algoritmo epsilon-decaimiento.

        :param k: Número de brazos.
        :param epsilon_initial: Probabilidad de exploración inicial.
        :param lambda_: Tasa de decaimiento de epsilon.
        :param epsilon_min: Valor mínimo que puede alcanzar epsilon.
        :raises ValueError: Si los parámetros no son válidos.
        """
        assert 0 <= epsilon_initial <= 1, "El parámetro epsilon_initial debe estar entre 0 y 1."
        assert lambda_ >= 0, "El parámetro lambda debe ser mayor o igual a 0."
        assert 0 <= epsilon_min <= 1, "El parámetro epsilon_min debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon_initial = epsilon_initial
        self.lambda_ = lambda_
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_initial
        self.t = 0

    def _calculate_decay(self) -> float:
        """
        Decaimiento Inversamente Proporcional
        """
        return self.epsilon_initial/(1 + self.lambda_ * self.t)

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-decaimiento.

        :return: índice del brazo seleccionado.
        """
        # Inicialización: seleccionar cada brazo al menos una vez antes de aplicar la política.
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited) > 0:
            return int(np.random.choice(unvisited))

        epsilon_t = max(self.epsilon_min, self._calculate_decay())

        if np.random.random() < epsilon_t:
            # Exploración: selecciona un brazo al azar
            chosen_arm = np.random.choice(self.k)
        else:
            # Explotación: selecciona el brazo con la recompensa promedio estimada más alta.
            # En caso de empate se rompe aleatoriamente.
            max_value = np.max(self.values)
            max_arms = np.where(self.values == max_value)[0]
            chosen_arm = np.random.choice(max_arms)

        self.t += 1

        return chosen_arm

    def reset(self):
        """
        Reinicia el estado del algoritmo, incluyendo el contador de tiempo t.
        Necesario para que el decaimiento de epsilon empiece desde cero en cada run.
        """
        super().reset()
        self.t = 0
        self.epsilon = self.epsilon_initial