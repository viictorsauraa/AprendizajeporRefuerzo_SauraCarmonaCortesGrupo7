# Plan de Lectura de Referencia: Partes 2 y 3

> Guía de qué leer, en qué orden y qué extraer de cada documento
> antes de implementar Q-Learning y Deep Q-Learning.

---

## 1. Lectura Obligatoria: Instrucciones de la Práctica

**Documento**: `Entrega_Parte_RL.pdf` (raíz del repositorio)

| Sección | Qué leer | Por qué |
|---|---|---|
| §2 y §3 | Criterios de evaluación e informe técnico | Define qué se evalúa: rigor metodológico, PDF ≤10 folios, reproducibilidad en Colab |
| §4.2 | Grupo 2. Aprendizaje en entornos complejos | Detalla tu parte exacta: comparar tabular (Q-Learning) vs aproximado (DQN) en ≥2 entornos |
| §5.2 | Uso básico de Gymnasium | Código base del bucle de episodios y esquema general del agente |
| §5.3 | Usar PyTorch para Definir una QNet | Plantilla exacta de la red neuronal que espera el profesor |
| §5.4 | Semillas fijas | Código de reproducibilidad para numpy, torch y gymnasium |

**Lo que debes extraer**:
- El esquema de la clase `Agent` con `__init__`, `get_action` y `update`
- El bucle `while not done` estándar de Gymnasium
- La plantilla `DQN_Network(nn.Module)` con capas lineales y ReLU
- El código de semilla fija para que los notebooks sean reproducibles

---

## 2. Base Teórica — Parte 2: Q-Learning Tabular

**Documento principal**: `Entornos_Complejos/docs/Transpa-DiferenciasTemporales.pdf`

| Sección | Qué leer | Por qué |
|---|---|---|
| Control TD Off-Policy | Teoría de Q-Learning | Explica por qué es off-policy: el target usa `max` sin importar la acción tomada |
| Pseudocódigo Q-Learning | Algoritmo completo | Es literalmente lo que debes traducir a `q_learning_agent.py` |
| (Opcional) Double Q-Learning | Sesgo de maximización | Cómo dividir la estimación en dos redes para evitar sobreestimar Q |

**La fórmula clave que debes conocer de memoria**:
```
Q(S,A) ← Q(S,A) + α * [R + γ * max_a Q(S',a) - Q(S,A)]
                                    ^^^^^^^^^
                              diferencia clave vs SARSA
                              (usa el max, no la acción tomada)
```

**Lo que debes extraer**:
- El pseudocódigo completo del algoritmo para implementar `update()` y `train()`
- La diferencia on-policy (SARSA) vs off-policy (Q-Learning): en Q-Learning el agente puede explorar libremente porque el target siempre usa la acción óptima

---

## 3. Base Teórica — Parte 3: Deep Q-Learning (DQN)

**Documento principal**: `Entornos_Complejos/docs/Transpa-ApproxControl.pdf`

| Sección | Qué leer | Por qué |
|---|---|---|
| Deep Q-Learning (DQN) | Arquitectura general | La red toma el estado como entrada y devuelve Q(s,a) para todas las acciones |
| Replay Buffer | Experience Replay | Guarda transiciones (s,a,r,s',done) y sampea batches aleatorios para romper correlaciones |
| Target Network | Red objetivo congelada | Segunda red cuyos pesos se actualizan cada C pasos para estabilizar el entrenamiento |
| (Recomendado) `Transpa-ApproxPrediccionOnPolicy.pdf` | Redes neuronales como funciones no lineales | Base de cómo el SGD actualiza pesos en contexto RL, si necesitas repasar teoría de redes |

**Los dos componentes que NO tiene SARSA semi-gradiente y DQN sí tiene**:

```
Replay Buffer D:
  - Capacidad máxima N (ej. 10.000 transiciones)
  - Guarda: (s, a, r, s', done)
  - Sampea minibatches aleatorios (ej. 64) para el update
  - Rompe la correlación entre transiciones consecutivas

Target Network Q_target:
  - Copia exacta de la red principal al inicio
  - Sus pesos se "congelan" durante C pasos
  - Se usa para calcular el target: y = r + γ * max_a Q_target(s',a)
  - Cada C pasos: w_target ← w  (hard update)
  - Alternativa: soft update w_target ← τ*w + (1-τ)*w_target
```

---

## 4. La "Biblia" para el Código de DQN — Paper Original

**Documento**: `Entornos_Complejos/docs/Playing_Atari_with_Deep_Reinforcement_Learning.pdf`

**Qué leer**: Ve directo al **Algorithm 1: Deep Q-learning with Experience Replay**

```
Algorithm 1 (resumen del paper):

Inicializar:
  D ← replay memory con capacidad N
  Q ← red principal con pesos aleatorios w
  Q_target ← red objetivo con pesos w_target = w

Para cada episodio:
  s_1 ← estado inicial del entorno

  Para cada paso t:
    Con prob ε: a_t ← acción aleatoria
    Si no:      a_t ← argmax_a Q(s_t, a; w)

    Ejecutar a_t, observar r_t, s_{t+1}, done

    Guardar (s_t, a_t, r_t, s_{t+1}, done) en D

    Sampear minibatch (s_j, a_j, r_j, s_{j+1}, done_j) de D

    Calcular target:
      y_j = r_j                                    si done_j
      y_j = r_j + γ * max_a Q_target(s_{j+1}, a)  si no done_j

    Actualizar w:
      Loss = (y_j - Q(s_j, a_j; w))²
      Descenso de gradiente sobre Loss

    Cada C pasos: w_target ← w
```

**Lo que debes extraer**:
- La estructura exacta del bucle de entrenamiento para `train()`
- Cuándo guardar en buffer vs cuándo actualizar (no se actualiza hasta que el buffer tiene suficientes muestras)
- Cómo calcular el target con la red objetivo (con `torch.no_grad()`)

**Documento adicional (opcional)**: `Deep_Reinforcement_Learning_with_Double_Q-learning.pdf`
- Implementa **Double DQN**: separa la selección de acción (red principal) de su evaluación (red objetivo)
- Target: `y = r + γ * Q_target(s', argmax_a Q(s', a; w); w_target)`
- Reduce la sobreestimación sistemática de Q-Learning clásico

---

## 5. Resumen del Plan de Acción

```
1. Leer §4.2 + §5.2 + §5.3 de Entrega_Parte_RL.pdf
   → Entender qué se pide y tener el esqueleto de código

2. Leer pseudocódigo en Transpa-DiferenciasTemporales.pdf
   → Implementar q_learning_agent.py
   → Crear QLearning_experiment.ipynb

3. Leer Transpa-ApproxControl.pdf + Algorithm 1 del paper de Atari
   → Implementar dqn_agent.py (ReplayBuffer + TargetNetwork + bucle)
   → Crear DeepQLearning_experiment.ipynb

4. (Opcional) Leer Double DQN paper
   → Añadir variante Double DQN al agente para comparar
```

---

## 6. Dónde están los documentos

```
EML/
├── Entrega_Parte_RL.pdf                                    ← Normas
└── Entornos_Complejos/
    └── docs/
        ├── Transpa-DiferenciasTemporales.pdf               ← Parte 2
        ├── Transpa-ApproxControl.pdf                       ← Parte 3
        ├── Transpa-ApproxPrediccionOnPolicy.pdf            ← Parte 3 (base redes)
        ├── Playing_Atari_with_Deep_Reinforcement_Learning.pdf  ← Parte 3 (paper)
        └── Deep_Reinforcement_Learning_with_Double_Q-learning.pdf  ← Opcional
```

> Si algún documento no está en `docs/`, añádelo antes de retomar la implementación.
