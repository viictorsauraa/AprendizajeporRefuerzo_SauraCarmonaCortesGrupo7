# Problema del Bandido de k-brazos

## Información

- **Alumnos:** Víctor Saura Meseguer, Guillermo Carmona Martínez, Francisco José Cortes Delgado
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** SauraCarmonaCortes (Grupo 7)

## Descripción

Estudio comparativo de algoritmos para el problema del **bandido de k-brazos** (*Multi-Armed Bandit*), un problema clásico de toma de decisiones secuenciales bajo incertidumbre. Se implementan y evalúan cinco algoritmos de tres familias distintas (ε-Greedy, UCB y Softmax) sobre tres tipos de distribuciones de recompensa (Normal, Binomial y Bernoulli), analizando el equilibrio exploración-explotación mediante métricas como el regret acumulado, la recompensa promedio y el porcentaje de selecciones óptimas.

## Estructura

```
k_brazos/
├── src/
│   ├── algorithms/         # Implementación de los algoritmos
│   │   ├── algorithm.py                        # Clase base abstracta
│   │   ├── epsilon_greedy.py                   # ε-Greedy
│   │   ├── epsilon_greedy_with_initialization.py  # ε-Greedy con inicialización
│   │   ├── epsilon_decaimiento.py              # ε-Decaimiento
│   │   ├── ucb1.py                             # UCB1
│   │   └── softmax.py                          # Softmax (Gibbs)
│   ├── arms/               # Implementación de los tipos de brazos
│   │   ├── arm.py                  # Clase base abstracta
│   │   ├── armnormal.py            # Brazo con distribución Normal
│   │   ├── armbinomial.py          # Brazo con distribución Binomial
│   │   ├── armbernoulli.py         # Brazo con distribución Bernoulli
│   │   └── bandit.py               # Máquina (conjunto de k brazos)
│   └── plotting/           # Funciones de visualización
│       └── plotting.py             # plot_regret, plot_optimal_selections, plot_arm_statistics, ...
├── docs/
│   └── Transpa-Bandido.pdf         # Transparencias de referencia del profesor
├── main.ipynb              # Punto de entrada: descripción del problema y enlaces a los estudios
├── bandit_experiment.ipynb                      # Estudio familia ε-Greedy sin inicialización (Normal)
├── bandit_experiment_Greedy_con_inicializacion.ipynb  # Estudio ε-Greedy con inicialización (Normal)
├── bandit_experiment_Normal.ipynb               # Comparativa completa — distribución Normal
├── bandit_experiment_Binomial.ipynb             # Comparativa completa — distribución Binomial
└── bandit_experiment_Bernoulli.ipynb            # Comparativa completa — distribución Bernoulli
```

## Instalación y Uso

Ejecutar en **Google Colab** (recomendado): abrir [`main.ipynb`](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/main.ipynb), que clona el repositorio e instala las dependencias automáticamente.

Para ejecución **local**, desde la raíz del repositorio:
```bash
pip install numpy matplotlib seaborn tqdm
```

Todas las ejecuciones usan `seed=42` para garantizar reproducibilidad.

## Guía de navegación

| Notebook | Contenido |
|---|---|
| [main.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/main.ipynb) | Punto de entrada. Configura el entorno e incluye enlaces a todos los estudios. |
| [bandit_experiment.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment.ipynb) | Estudio de ε-Greedy (ε = 0, 0.01, 0.1) sin inicialización sobre brazos Normal. |
| [bandit_experiment_Greedy_con_inicializacion.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment_Greedy_con_inicializacion.ipynb) | Mismo estudio con inicialización round-robin. Compara el efecto de visitar todos los brazos antes de explotar. |
| [bandit_experiment_Normal.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment_Normal.ipynb) | Comparativa ε-Greedy, ε-Decaimiento, UCB1, Softmax sobre distribución Normal. |
| [bandit_experiment_Binomial.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment_Binomial.ipynb) | Comparativa de algoritmos sobre distribución Binomial B(n, p). |
| [bandit_experiment_Bernoulli.ipynb](https://colab.research.google.com/github/viictorsauraa/AprendizajeporRefuerzo_SauraCarmonaCortesGrupo7/blob/main/k_brazos/bandit_experiment_Bernoulli.ipynb) | Comparativa de algoritmos sobre distribución Bernoulli (recompensas binarias {0, 1}). |

## Tecnologías Utilizadas

- **Python 3**
- **NumPy** — cálculo numérico y generación de distribuciones
- **Matplotlib / Seaborn** — visualización de resultados
- **tqdm** — barras de progreso
- **Google Colab** — entorno de ejecución en la nube
