"""
Module: agents/sarsa_sg_agent.py
Description: Implementación del algoritmo SARSA semi gradiente para entornos de Gymnasium.

"""

import numpy as np
import gymnasium as gym
from agents.gymnasium_agent import GymnasiumAgent
from tqdm import tqdm
import torch
import torch.nn as nn


class DQN_Network(nn.Module):
    """
    Redes neuronales para aproximar la función de valor Q en el agente SARSA semi gradiente.
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

class SARSASGAgent(GymnasiumAgent):

    def __init__(self, env: gym.Env, epsilon: float, discount_factor: float, decay: bool, alpha: float, weights_path=None, hidden_size=64, num_hidden_layers=2, use_cpu = True):
        """
        Inicializa el agente SARSA con un entorno de Gymnasium.
        :param alpha: Tasa de aprendizaje para la actualización de valores Q.
        """
        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.decay = decay

        self.device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
        print(f"Entrenando en: {self.device}")
        self.alpha = alpha
        self.network = DQN_Network(env.action_space.n, env.observation_space.shape[0], hidden_size=hidden_size, num_hidden_layers=num_hidden_layers).to(self.device)
        if weights_path is not None:
            self.network.load_state_dict(torch.load(weights_path))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)
    
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
            # Convertimos y enviamos a la GPU en un solo paso
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.network(obs_tensor)
            best_action = torch.argmax(q_values).item()
            return best_action

    def update(self, obs, action, reward, next_obs, next_action, terminated, truncated, info):
        """
        Actualiza los valores Q usando la regla de actualización SARSA semi gradiente.
        """
        # Convertimos estados a tensores
        s_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        s_next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        # calculamos q^(St, At, wt)

        q_values = self.network(s_t)
        q_prediction = q_values[action] # Valor estimado para la acción tomada

        # calculo de Ut = Rt+1 + γ * q^(St+1, At+1, wt)
        # Usamos no_grad() porque es sarsa semigradiente, el objetivo se trata como constante
        with torch.no_grad(): 
            if terminated or truncated:
                target_u = torch.tensor(reward, dtype=torch.float32, device=self.device) # Ut = Rt+1
            else:
                q_next = self.network(s_next_t)
                target_u = reward + self.discount_factor * q_next[next_action] # Ut = Rt+1 + γ * q^(St+1, At+1, wt)


        # calculo del error y del gradiente: [Ut - q^] * ∇q^
        # En PyTorch, la pérdida (Loss) es 0.5 * (Target - Predicción)^2
        # Al derivar (backward), obtenemos exactamente: -(Ut - q^) * ∇q^
        loss = 0.5 * (target_u - q_prediction)**2

        self.optimizer.zero_grad()
        loss.backward() # Aquí PyTorch calcula el gradiente: ∇q^(St, At, wt)

        # actualización final: wt+1 = wt + α * [Error * Gradiente]

        self.optimizer.step() # El optimizador aplica α y actualiza los pesos w

        self.list_losses.append(loss.item())
    def train(self, num_episodes):
        env = self.env
        env.reset(seed=2024)
        for t in tqdm(range(num_episodes)):
            obs, info = env.reset()

            step_display = max(1, num_episodes // 10)
            done = False

            action = self.get_action(obs)
            episode_length = 0
            episode_reward = 0

            while not done:
                next_obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                next_action = None
                if not done:
                    next_action = self.get_action(next_obs)

                self.update(obs,action,reward,next_obs,next_action,terminated, truncated, info)

                obs = next_obs
                action = next_action

                episode_reward += reward
                episode_length += 1
            
            if self.decay:
                self.epsilon = max(0.01, self.epsilon * 0.995) # Cambiamos formula de dacay para que se adapte a tener menos episodios

            self.update_stats(episode_reward, episode_length)

            if (t + 1) % step_display == 0:
                list_stats, _, list_losses = self.stats()
                print(f"success: {list_stats[-1]:.4f}, epsilon: {self.epsilon:.4f}, loss: {list_losses[-1]:.4f}")

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