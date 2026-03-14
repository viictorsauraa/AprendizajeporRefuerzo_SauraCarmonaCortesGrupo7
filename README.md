# Aprendizaje por Refuerzo

## Información

- **Alumnos:** Víctor Saura Meseguer, Guillermo Carmona Martínez, Francisco José Cortes Delgado
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** SauraCarmonaCortes (Grupo 7)
- **Repositorio:** https://github.com/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7

## Descripción

Este repositorio recoge el trabajo práctico de la asignatura, organizado en tres partes:

- **Parte 1 — Bandido de k-brazos:** Estudio comparativo de algoritmos para el problema del *Multi-Armed Bandit* (ε-Greedy, ε-Decaimiento, UCB1, Softmax) sobre distribuciones Normal, Binomial y Bernoulli.
- **Parte 2 — Métodos tabulares:** Implementación y comparación de Monte Carlo (on/off-policy), SARSA y Q-Learning en los entornos discretos Taxi-v3 y CliffWalking-v0.
- **Parte 3 — Métodos aproximados:** SARSA semi-gradiente y Deep Q-Learning con Experience Replay en el entorno continuo LunarLander-v3.

## Estructura

```
EML/
├── main.ipynb                  # Punto de entrada general del proyecto
├── k_brazos/                   # Parte 1: Bandido de k-brazos
│   ├── src/                    # Algoritmos, brazos y visualización
│   ├── docs/                   # Transparencias de referencia
│   ├── main.ipynb              # Punto de entrada de la Parte 1
│   ├── bandit_experiment.ipynb
│   ├── bandit_experiment_Normal.ipynb
│   ├── bandit_experiment_Binomial.ipynb
│   ├── bandit_experiment_Bernoulli.ipynb
│   ├── bandit_experiment_Greedy_con_inicializacion.ipynb
│   └── README.md
└── Entornos_Complejos/         # Partes 2 y 3: RL tabular y aproximado
    ├── src/                    # Agentes y visualización
    ├── docs/                   # Transparencias de referencia
    ├── tests/                  # Notebooks auxiliares y de validación
    ├── weights/                # Pesos de redes entrenadas (.pth)
    ├── metrics/                # Métricas guardadas (.npz)
    ├── main.ipynb              # Punto de entrada de las Partes 2 y 3
    ├── MonteCarloTodasLasVisitas.ipynb
    ├── MonteCarlo_experiment.ipynb
    ├── MonteCarloOffPolicy.ipynb
    ├── SARSA_experiment.ipynb
    ├── SARSA_experiment_CLIFF.ipynb
    ├── Qlearning_experiment.ipynb
    ├── Qlearning_experiment_CLIFF.ipynb
    ├── SARSA_SG_experiment.ipynb
    ├── DeepQLearning_experiment.ipynb
    └── README.md
```

## Instalación y Uso

La forma recomendada de ejecutar el proyecto es a través de **Google Colab** sin instalación local:

1. Abrir [`main.ipynb`](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/main.ipynb) en Colab.
2. Ejecutar la celda de preparación del entorno (clona el repositorio e instala dependencias).
3. Navegar a cualquier notebook desde los enlaces de la tabla inferior.

Cada notebook de experimentos también es **autocontenido**: incluye sus propias celdas de clonado del repositorio e instalación de dependencias, por lo que puede abrirse y ejecutarse directamente en Colab sin pasar por `main.ipynb`.

Para ejecución **local**:
```bash
git clone https://github.com/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7.git
cd AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7
pip install -r requirements.txt
```

## Guía de navegación

### Parte 1 — Bandido de k-brazos

| Notebook | Descripción |
|---|---|
| [main.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/main.ipynb) | Punto de entrada de la Parte 1 |
| [bandit_experiment.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment.ipynb) | Estudio ε-Greedy sobre brazos Normal |
| [bandit_experiment_Greedy_con_inicializacion.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment_Greedy_con_inicializacion.ipynb) | ε-Greedy con inicialización round-robin |
| [bandit_experiment_Normal.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment_Normal.ipynb) | Comparativa completa — distribución Normal |
| [bandit_experiment_Binomial.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment_Binomial.ipynb) | Comparativa completa — distribución Binomial |
| [bandit_experiment_Bernoulli.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment_Bernoulli.ipynb) | Comparativa completa — distribución Bernoulli |

### Partes 2 y 3 — Entornos Complejos

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
- **NumPy / Matplotlib / Seaborn** — cálculo numérico y visualización
- **tqdm** — barras de progreso
- **Gymnasium** — entornos de simulación (FrozenLake-v1, Taxi-v3, CliffWalking-v0, LunarLander-v3)
- **PyTorch** — redes neuronales para SARSA-SG y DQN
- **Google Colab** — entorno de ejecución en la nube
