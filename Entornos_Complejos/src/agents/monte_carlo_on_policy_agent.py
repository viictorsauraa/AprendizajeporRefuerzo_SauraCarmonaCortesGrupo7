"""
Module: agents/monte_carlo_on_policy_agent.py
Description: Implementación del algoritmo Monte Carlo On-Policy (todas las visitas)
             para entornos de Gymnasium. Algoritmo 3 de Sutton & Barto (2018).
"""
import numpy as np
import gymnasium as gym
from agents.gymnasium_agent import GymnasiumAgent
from tqdm import tqdm


class MonteCarloOnPolicyAgent(GymnasiumAgent):
    """
    Agente Monte Carlo On-Policy con criterio de todas las visitas.
    Implementa el Algoritmo 3 de Sutton & Barto (2018).

    La política de exploración y la política objetivo son la misma (epsilon-greedy).
    Actualiza Q con media incremental: Q(s,a) += (1/n(s,a)) * (G - Q(s,a)).
    Los retornos se calculan hacia atrás al final de cada episodio.
    """

    def __init__(self, env: gym.Env, epsilon: float = 0.2,
                 discount_factor: float = 1.0, decay: bool = False):
        super().__init__(env, epsilon, discount_factor, decay)
        self._episode = []   # (state, action, reward)

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Acumula (s, a, r) durante el episodio.
        Al terminar: calcula retornos G hacia atrás y actualiza Q (todas las visitas).
        Media incremental: Q(s,a) += (1/n_visits) * (G - Q(s,a)).
        """
        self._episode.append((obs, action, reward))

        if terminated or truncated:
            G = 0.0
            for (s, a, r) in reversed(self._episode):
                G = r + self.discount_factor * G
                self.num_visited[s, a] += 1.0
                alpha = 1.0 / self.num_visited[s, a]
                self.q_values[s, a] += alpha * (G - self.q_values[s, a])

            episode_reward = sum(r for _, _, r in self._episode)
            episode_length = len(self._episode)
            self.update_stats(episode_reward, episode_length)

            if self.decay:
                self.epsilon = min(1.0, 1000.0 / (self._episode_count + 1))

            self._episode = []

    def train(self, num_episodes: int):
        """
        Entrena al agente durante num_episodes episodios usando el bucle
        episódico estándar de Gymnasium (sección 5.2).
        """
        env = self.env
        step_display = max(1, num_episodes // 10)

        for t in tqdm(range(num_episodes)):
            obs, info = env.reset()
            done = False

            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.update(obs, action, next_obs, reward, terminated, truncated, info)
                done = terminated or truncated
                obs = next_obs

            if (t + 1) % step_display == 0:
                list_stats, _ = self.stats()
                print(f"success: {list_stats[-1]:.4f}, epsilon: {self.epsilon:.4f}")
