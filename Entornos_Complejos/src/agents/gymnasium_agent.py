"""
Module: agents/gymnasium_agent.py
Description: Contiene la implementación de un agente para entornos Gymnasium.

"""

from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym

class GymnasiumAgent(ABC):
    def __init__(self, env: gym.Env, epsilon: float, discount_factor: float, decay: bool ):
        """
        Inicializa el agente con un entorno de Gymnasium.
        :param env: Entorno de Gymnasium.
        :param epsilon: Parámetro de exploración para la política epsilon-greedy.
        :param discount_factor: Factor de descuento para la actualización de valores.
        :param decay: Indica si se debe aplicar decaimiento a epsilon.
        """
        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.decay = decay

        # Estructuras de datos para almacenar los valores Q y el número de visitas a cada estado-acción
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.num_visited = np.zeros([env.observation_space.n, env.action_space.n])

        # Estadísticas
        self._episode_count = 0
        self._total_reward = 0.0
        self._list_stats = [0.0]
        self._list_lengths = []

    def get_action(self, obs):
        """
        Selecciona una acción basada en la política epsilon-greedy.
        :param obs: Observación actual del entorno.
        :return: Acción seleccionada.
        """
        actions = np.ones(self.env.action_space.n) * self.epsilon / self.env.action_space.n

        best_action = np.argmax(self.q_values[obs])

        actions[best_action] += 1 - self.epsilon

        return np.random.choice(np.arange(self.env.action_space.n), p=actions)
    
    @abstractmethod
    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Actualiza los valores Q basados en la transición observada.
        :param obs: Observación actual del entorno.
        :param action: Acción tomada.
        :param next_obs: Observación siguiente después de tomar la acción.
        :param reward: Recompensa obtenida.
        :param terminated: Indica si el episodio ha terminado.
        :param truncated: Indica si el episodio ha sido truncado.
        :param info: Información adicional del entorno.
        """
        raise NotImplementedError("Este método debe ser implementado por la subclase.")

    @abstractmethod
    def train(self, num_episodes: int):
        """
        Entrena al agente durante un número específico de episodios.
        :param num_episodes: Número de episodios para entrenar al agente.
        """
        raise NotImplementedError("Este método debe ser implementado por la subclase.")
    
    def __pi_star_from_Q__(self, Q):
        done = False
        env = self.env
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset()
        actions = ""
        while not done:
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state, action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return pi_star, actions
    
    def pi_star(self):
        """Política óptima greedy a partir de Q aprendida. Delega en pi_star_from_Q."""
        return self.__pi_star_from_Q__(self.q_values)
    
    def stats(self):
        """
        Retorna los resultados estadísticos y de evolución:
          - list_stats:   proporción acumulada de recompensas por episodio
          - list_lengths: longitud (en pasos) de cada episodio
        """
        return self._list_stats, self._list_lengths

    def update_stats(self, episode_reward, episode_length):
        """
        Actualiza las estadísticas del agente después de cada episodio.
        :param episode_reward: Recompensa total obtenida en el episodio.
        :param episode_length: Duración del episodio en pasos.
        """
        self._episode_count += 1
        self._total_reward += episode_reward
        self._list_lengths.append(episode_length)
        self._list_stats.append(self._total_reward / self._episode_count)   