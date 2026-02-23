"""
Module: agents/monte_carlo_off_policy_agent.py
Description: Implementación del algoritmo Monte Carlo Off-Policy con muestreo
             por importancia ponderado. Algoritmo 6 de Sutton & Barto (2018).
"""
import numpy as np
import gymnasium as gym
from agents.gymnasium_agent import GymnasiumAgent
from tqdm import tqdm


class MonteCarloOffPolicyAgent(GymnasiumAgent):
    """
    Agente Monte Carlo Off-Policy con muestreo por importancia ponderado.
    Implementa el Algoritmo 6 de Sutton & Barto (2018).

    - Política de comportamiento b: epsilon-greedy sobre Q (genera los episodios).
    - Política objetivo pi: greedy sobre Q (la que se mejora).

    Bucle de retropropagación al finalizar cada episodio:
      C(s,a) += W
      Q(s,a) += (W/C(s,a)) * (G - Q(s,a))
      Si A_t != argmax Q[s]: break  (pi es determinista, W se haría 0)
      W *= 1/b(a|s)

    C(s,a) acumula los pesos de importancia (denominador del IS ponderado).

    Parámetro q_init:
      Por defecto None → Q = 0. Si se pasa un valor positivo (e.g. 0.01),
      Q se inicializa con valores aleatorios uniformes en [0, q_init]. Rompe el
      sesgo de argmax hacia acción 0 en entornos grandes con recompensas dispersas.
    """

    def __init__(self, env: gym.Env, epsilon: float = 0.3,
                 discount_factor: float = 1.0, decay: bool = False,
                 q_init: float = None):
        super().__init__(env, epsilon, discount_factor, decay)

        if q_init is not None:
            rng = np.random.default_rng(42)
            self.q_values = rng.uniform(0, q_init, self.q_values.shape)

        self.C = np.zeros_like(self.q_values)  # acumulador de pesos IS ponderado
        self._episode = []                      # (state, action, reward, b_prob)

    def _b_prob(self, obs, action):
        """Probabilidad de la política de comportamiento b(a|s) (epsilon-greedy)."""
        nA = self.env.action_space.n
        greedy_a = np.argmax(self.q_values[obs])
        if action == greedy_a:
            return 1.0 - self.epsilon + self.epsilon / nA
        return self.epsilon / nA

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Acumula (s, a, r, b_prob) durante el episodio.
        b_prob se calcula ANTES de cualquier actualización de Q.
        Al terminar aplica el Algoritmo 6:
          - Retornos G hacia atrás
          - Actualización ponderada con C y W
          - Break cuando acción != argmax Q[s] (pi greedy determina W=0)
        """
        bp = self._b_prob(obs, action)
        self._episode.append((obs, action, reward, bp))

        if terminated or truncated:
            G = 0.0
            W = 1.0

            for (s, a, r, bp) in reversed(self._episode):
                G = r + self.discount_factor * G
                self.C[s, a] += W
                self.q_values[s, a] += (W / self.C[s, a]) * (G - self.q_values[s, a])

                if a != np.argmax(self.q_values[s]):
                    break

                W *= 1.0 / bp

            episode_reward = sum(r for _, _, r, _ in self._episode)
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
