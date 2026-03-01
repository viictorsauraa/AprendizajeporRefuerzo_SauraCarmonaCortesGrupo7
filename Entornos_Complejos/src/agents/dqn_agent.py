"""
Module: agents/dqn_agent.py
Description: Implementación del algoritmo Deep Q Learning Network para entornos de Gymnasium.

"""
import random
from collections import deque

import numpy as np
import gymnasium as gym
from agents.gymnasium_agent import GymnasiumAgent
from tqdm import tqdm
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# DQN_Network: Red Principal es igual que la de SARSA semi gradiente, no obstante decidimos copiarla y pegarla en vez de
# importarla por si queremos hacer modificaciones específicas para DQN en el futuro. Además, así evitamos dependencias entre agentes.
class DQN_Network(nn.Module):
    """
    Redes neuronales para aproximar la función de valor Q en el agente DQN.
    """
    def __init__(self, num_actions, input_dim, hidden_size=64, num_hidden_layers=2):
        super(DQN_Network, self).__init__()
        
        layers = []
        
        # Capa de entrada
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU(inplace=True))
        
        # Número de capas ocultas
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
        
        # Capa de salida
        layers.append(nn.Linear(hidden_size, num_actions))
        
        self.FC = nn.Sequential(*layers)
        
        for module in self.FC:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        return self.FC(x)

# DQNAgent: Gymnasium Agent
class DQNAgent(GymnasiumAgent):

    def __init__(self, env: gym.Env, epsilon: float, discount_factor: float, decay: bool, alpha: float,
                 buffer_capacity: int, batch_size: int, target_update_freq: int,
                 weights_path=None, hidden_size=64, num_hidden_layers=2, use_cpu=True):
        """
        Inicializa el agente DQN con un entorno de Gymnasium.
        :param alpha: Tasa de aprendizaje para la actualización de valores Q.
        :param buffer_capacity: Capacidad máxima del Replay Buffer.
        :param batch_size: Número de transiciones por minibatch en cada actualización.
        :param target_update_freq: Número de pasos entre cada copia w- <- w.
        """
        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.decay = decay

        self.device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
        print(f"Entrenando en: {self.device}")
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # ReplayBuffer: almacena tuplas (obs, action, reward, next_obs, done)
        # y permite samplear minibatches aleatorios para romper correlaciones temporales.
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # DeepQNetwork (Red Principal): red neuronal con pesos w que, dado un vector
        # de estado continuo (las 8 variables), devuelve Q(s, a; w) — el valor Q
        # estimado para cada una de las 4 posibles acciones. Guiará a la política
        # epsilon-greedy a elegir la mejor acción.
        self.network = DQN_Network(env.action_space.n, env.observation_space.shape[0], hidden_size=hidden_size, num_hidden_layers=num_hidden_layers).to(self.device)
        if weights_path is not None:
            self.network.load_state_dict(torch.load(weights_path))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)

        # Target Network (Red Objetivo): copia congelada de la red principal con pesos w-.
        # Calcula Q(s', a; w-) para construir el target y = r + gamma * max_a Q(s', a; w-).
        # Sus pesos se copian desde w cada target_update_freq pasos, proporcionando
        # un blanco fijo que estabiliza el entrenamiento.
        self.target_network = DQN_Network(env.action_space.n, env.observation_space.shape[0], hidden_size=hidden_size, num_hidden_layers=num_hidden_layers).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Contador global de pasos para controlar cuándo actualizar la target network
        self._step_count = 0

        # Estadísticas
        self._episode_count = 0
        self._total_reward = 0.0
        self._list_stats = [0.0]
        self._list_lengths = []
        self.list_losses = []

    def get_action(self, obs):
        """
        Selecciona una acción basada en la política epsilon-greedy.
        :param obs: Observación actual del entorno.
        :return: Acción seleccionada.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.network(obs_tensor)
            return torch.argmax(q_values).item()
        
    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        """
        1. Almacena la transición en el Replay Buffer.
        2. Si el buffer tiene suficientes muestras, samplea un minibatch.
        3. Calcula el target y = r + gamma * max_a Q(s', a; w-) con la Target Network.
        4. Calcula Q(s, a; w) con la Red Principal y la loss MSE.
        5. Realiza un paso de optimización para actualizar los pesos w.
        6. Cada target_update_freq pasos copia w- <- w.
        """
        # Almacenamos la transición en el Replay Buffer
        self.replay_buffer.store(obs, action, reward, next_obs, terminated or truncated)

        # Solo actualizamos si tenemos suficientes muestras para un minibatch
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sampleamos un minibatch aleatorio de transiciones del Replay Buffer
        batch = self.replay_buffer.sample(self.batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(* batch)
        obs_batch = torch.as_tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(np.array(action_batch), dtype=torch.int64, device=self.device)
        reward_batch = torch.as_tensor(np.array(reward_batch), dtype=torch.float32, device=self.device)
        next_obs_batch = torch.as_tensor(np.array(next_obs_batch), dtype=torch.float32, device=self.device)
        done_batch = torch.as_tensor(np.array(done_batch), dtype=torch.float32, device=self.device)
        
        # Calculamos el target y = r + gamma * max_a Q(s', a; w-) con la Target Network
        with torch.no_grad():
            # max_a Q(next_obs, a; w-) usando la Target Network
            q_targets = self.target_network(next_obs_batch)      # Q para las 4 acciones
            max_q_targets = q_targets.max(dim=1).values          # max sobre las 4 acciones
            # r si es done, sino r + gamma * max_a Q(next_obs, a; w-)
            y = reward_batch + (1.0 - done_batch) * self.discount_factor * max_q_targets

        # Calculamos Q(s, a; w) para las acciones tomadas con la Red Principal
        q_values = self.network(obs_batch)
        q_values = q_values.gather(dim=1, index=action_batch.unsqueeze(1))

        # Calculamos la pérdida (MSE entre Q(s, a; w) y el target Q(s', a; w-))
        loss = torch.mean((q_values.squeeze(1) - y) ** 2)
        # Actualizamos los parámetros de la red principal
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.list_losses.append(loss.item())

        # Actualizamos la Target Network cada target_update_freq pasos
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def train(self, num_episodes):
        env = self.env
        env.reset(seed=2024)
        step_display = max(1, num_episodes // 10)

        for t in tqdm(range(num_episodes)):
            obs, info = env.reset()
            done = False
            episode_length = 0
            episode_reward = 0

            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                self.update(obs, action, reward, next_obs, terminated, truncated, info)

                obs = next_obs
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            if self.decay:
                self.epsilon = max(0.01, self.epsilon * 0.995)

            self.update_stats(episode_reward, episode_length)

            if (t + 1) % step_display == 0:
                list_stats, _, list_losses = self.stats()
                loss_val = list_losses[-1] if list_losses else 0.0
                print(f"success: {list_stats[-1]:.4f}, epsilon: {self.epsilon:.4f}, loss: {loss_val:.4f}")

    def stats(self):
        """
        Retorna los resultados estadísticos y de evolución:
          - list_stats:   proporción acumulada de recompensas por episodio
          - list_lengths: longitud (en pasos) de cada episodio
          - list_losses:  lista de pérdidas (loss) por paso durante el entrenamiento
        """
        return self._list_stats, self._list_lengths, self.list_losses
    
    def save_model(self, path):
        """
        Guarda el modelo de la red neuronal en un archivo.
        :param path: Ruta del archivo donde se guardará el modelo.
        """
        torch.save(self.network.state_dict(), path)

