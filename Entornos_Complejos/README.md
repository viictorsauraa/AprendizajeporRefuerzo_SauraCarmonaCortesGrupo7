# Aprendizaje en Entornos Complejos

## Información

- **Alumnos:** Víctor Saura Meseguer, Guillermo Carmona Martínez, Francisco José Cortes Delgado
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** SauraCarmonaCortes (Grupo 7)

## Descripción

Estudio comparativo de algoritmos de Aprendizaje por Refuerzo aplicados a entornos de Gymnasium. Se cubren dos grupos de técnicas: **métodos tabulares** (Monte Carlo, SARSA, Q-Learning) sobre el entorno discreto Taxi-v3 y Cliff-Walking, y **métodos con aproximación de función** (SARSA semi-gradiente, Deep Q-Learning) sobre el entorno continuo LunarLander-v3. La necesidad de pasar a métodos aproximados queda justificada por el espacio de observación continuo de LunarLander-v3, que imposibilita una representación tabular.

## Estructura

```
Entornos_Complejos/
├── src/
│   ├── agents/                             # Implementación de los agentes
│   │   ├── gymnasium_agent.py              # Clase base abstracta (ε-greedy, stats)
│   │   ├── monte_carlo_on_policy_agent.py  # MC On-Policy todas las visitas (Alg. 3 S&B)
│   │   ├── monte_carlo_off_policy_agent.py # MC Off-Policy importancia ponderada (Alg. 6 S&B)
│   │   ├── sarsa_agent.py                  # SARSA — TD on-policy tabular
│   │   ├── q_learning_agent.py             # Q-Learning — TD off-policy tabular
│   │   ├── sarsa_sg_agent.py               # SARSA semi-gradiente con red neuronal
│   │   └── dqn_agent.py                    # DQN con Replay Buffer y Target Network
│   └── plotting/
│       └── plotting.py                     # Funciones de visualización
├── docs/
│   ├── Transpa-MonteCarlo.pdf              # Teoría: Monte Carlo
│   ├── Transpa-DiferenciasTemporales.pdf   # Teoría: SARSA y Q-Learning
│   ├── Transpa-ApproxPrediccionOnPolicy.pdf # Teoría: aproximación on-policy
│   └── Transpa-ApproxControl.pdf           # Teoría: control con aproximación (SARSA-SG, DQN)
├── tests/                                  # Notebooks de validación y ejemplos auxiliares
│   ├── evaluacion_bellman.ipynb            # Verificación ecuaciones de Bellman
│   └── EjemploGeneracionVideos.ipynb       # Wrapper para generar vídeos de episodios
├── weights/                                # Pesos guardados de redes entrenadas (.pth)
├── metrics/                                # Métricas guardadas (.npz) para reproducir gráficas
├── main.ipynb                              # Punto de entrada: descripción y enlaces a todos los estudios
├── MonteCarloTodasLasVisitas.ipynb         # MC On-Policy todas las visitas — FrozenLake-v1 (4×4 y 8×8)
├── MonteCarlo_experiment.ipynb             # MC On-Policy y Off-Policy — Taxi-v3
├── MonteCarloOffPolicy.ipynb               # MC Off-Policy en profundidad — FrozenLake-v1 (4×4 y 8×8)
├── SARSA_experiment.ipynb                  # SARSA tabular — Taxi-v3
├── SARSA_experiment_CLIFF.ipynb            # SARSA tabular — CliffWalking-v0
├── Qlearning_experiment.ipynb              # Q-Learning tabular — Taxi-v3
├── Qlearning_experiment_CLIFF.ipynb        # Q-Learning tabular — CliffWalking-v0
├── SARSA_SG_experiment.ipynb               # SARSA semi-gradiente — LunarLander-v3
└── DeepQLearning_experiment.ipynb          # Deep Q-Learning — LunarLander-v3
```

## Instalación y Uso

Ejecutar en **Google Colab** (recomendado): abrir [`main.ipynb`](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/main.ipynb), que clona el repositorio, instala `gymnasium[box2d]`, `torch`, `matplotlib` y `tqdm`, y enlaza a todos los notebooks de experimentos.

Para ejecución **local**, desde la raíz del repositorio:
```bash
pip install gymnasium[box2d] torch matplotlib tqdm
```

## Guía de navegación

| Notebook | Algoritmo | Entorno |
|---|---|---|
| [main.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/main.ipynb) | — | Punto de entrada |
| [MonteCarloTodasLasVisitas.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/MonteCarloTodasLasVisitas.ipynb) | MC On-Policy todas las visitas | FrozenLake-v1 (4×4 y 8×8) |
| [MonteCarlo_experiment.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/MonteCarlo_experiment.ipynb) | MC On-Policy + Off-Policy | Taxi-v3 |
| [MonteCarloOffPolicy.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/MonteCarloOffPolicy.ipynb) | MC Off-Policy (importancia ponderada) | FrozenLake-v1 (4×4 y 8×8) |
| [SARSA_experiment.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/SARSA_experiment.ipynb) | SARSA | Taxi-v3 |
| [SARSA_experiment_CLIFF.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/SARSA_experiment_CLIFF.ipynb) | SARSA | CliffWalking-v0 |
| [Qlearning_experiment.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/Qlearning_experiment.ipynb) | Q-Learning | Taxi-v3 |
| [Qlearning_experiment_CLIFF.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/Qlearning_experiment_CLIFF.ipynb) | Q-Learning | CliffWalking-v0 |
| [SARSA_SG_experiment.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/SARSA_SG_experiment.ipynb) | SARSA semi-gradiente | LunarLander-v3 |
| [DeepQLearning_experiment.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/Entornos_Complejos/DeepQLearning_experiment.ipynb) | Deep Q-Learning | LunarLander-v3 |

## Tecnologías Utilizadas

- **Python 3**
- **NumPy / Matplotlib** — cálculo numérico y visualización
- **Gymnasium** — entornos de simulación (Taxi-v3, LunarLander-v3)
- **PyTorch** — redes neuronales para SARSA-SG y DQN
- **Google Colab** — entorno de ejecución en la nube

Todas las ejecuciones usan `seed=2024` para garantizar reproducibilidad.
