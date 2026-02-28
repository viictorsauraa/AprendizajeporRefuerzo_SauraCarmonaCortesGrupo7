"""
Module: agents/q_learning_agent.py
Description: Implementación del algoritmo Q-Learning para entornos de Gymnasium.

"""
import numpy as np
import gymnasium as gym
from agents.gymnasium_agent import GymnasiumAgent
from tqdm import tqdm

class QLearningAgent(GymnasiumAgent):
    """
        Q-Learning agent.

         Q(S, A) = Q(S, A) + alpha * (R + gamma * max_a Q(S', a) - Q(S, A))
        """
    def __init__(self, env: gym.Env, epsilon: float, discount_factor: float, decay: bool, alpha: float):
        """
        Inicializa el agente Q-Learning con un entorno de Gymnasium.
        :param alpha: Tasa de aprendizaje para la actualización de valores Q.
        """
        super().__init__(env, epsilon, discount_factor, decay)
        self.alpha = alpha
    
    # Para get_action se usará la política epsilon-greedy

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Actualiza los valores Q usando la regla de actualización Q-Learning.
        """
        alpha = self.alpha
        discount_factor = self.discount_factor

        # DT = R + gamma * max_a Q(S', a) - Q(S, A)
        temporal_difference = reward + discount_factor * (self.q_values[next_obs, np.argmax(self.q_values[next_obs])]) - self.q_values[obs,action]

        # Q(S, A) = Q(S, A) + alpha * DT
        self.q_values[obs,action] += alpha * temporal_difference

    
    def train(self, num_episodes):
        """
        Entrena al agente durante un número específico de episodios utilizando el algoritmo Q-Learning.
        """
        env = self.env
        env.reset(seed=2024)
        for t in tqdm(range(num_episodes)):
            # Reiniciar el entorno al inicio de cada episodio
            obs, info = env.reset()
            step_display = max(1, num_episodes // 10)
            done = False
            
            episode_length = 0
            episode_reward = 0

            while not done:
                # Seleccionar la acción inicial utilizando la política epsilon-greedy
                action = self.get_action(obs)

                # Tomamos la acción y observamos el resultado
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Actualizamos los valores Q utilizando la regla de actualización Q-Learning
                self.update(obs, action, next_obs, reward, terminated, truncated, info)

                # Actualizamos la observación actual a la siguiente observación
                obs = next_obs

                done = terminated or truncated
                episode_length += 1
                episode_reward += reward
        
            # Decaimiento de epsilon
            if self.decay:
                self.epsilon = min(1.0, 1000.0 / (self._episode_count + 1))

            self.update_stats(episode_reward, episode_length)

            if (t + 1) % step_display == 0:
                list_stats, _ = self.stats()
                print(f"success: {list_stats[-1]:.4f}, epsilon: {self.epsilon:.4f}")