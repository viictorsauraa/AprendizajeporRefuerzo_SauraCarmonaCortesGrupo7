# Contexto de Trabajo: Parte 2 (Q-Learning) y Parte 3 (Deep Q-Learning)

> Documento de referencia para retomar la práctica. Resume el estado actual,
> los patrones del código, los requisitos del profesor y el plan de implementación.

---

## ⚠️ ADVERTENCIA CRÍTICA: ORIGINALIDAD (leer antes que nada)

El profesor lo dice explícitamente y en negrita en el enunciado:

> *"No se busca reproducir un experimento típico de internet. La originalidad metodológica
> forma parte de la evaluación."*
>
> *"Reproducir algo muy similar a lo que aparece en tutoriales públicos (como los de Gymnasium)
> puede ser considerado falta grave de originalidad académica, incluso aunque no haya copia
> literal de código."*

### Qué significa esto en la práctica

**Lo que sí se evalúa:**
- Qué problema planteas y por qué tiene interés
- Qué diseño experimental defines (hipótesis, variables, métricas)
- Cómo justificas técnicamente cada decisión
- Cómo analizas los resultados (no solo mostrarlos, interpretarlos)
- Qué conclusiones extraes y cómo las argumentas

**Lo que NO basta con hacer:**
- Coger CartPole o LunarLander, entrenar DQN, mostrar que la recompensa sube
- Usar los mismos entornos con los mismos hiperparámetros que aparecen en tutoriales
- Reproducir el "experimento estándar de internet" aunque el código sea propio
- Cambiar solo una media que aparece en un GitHub por otra media

### Lo que implica para el diseño del experimento

Antes de escribir código hay que tener una **pregunta experimental propia**. Ejemplos de preguntas con criterio:
- ¿Cómo afecta el tamaño del Replay Buffer a la estabilidad de DQN en entornos con recompensa esparsa?
- ¿En qué punto exacto (número de estados) Q-Learning tabular deja de ser viable y DQN lo supera?
- ¿Qué diferencia práctica introduce la Target Network frente a no usarla (= SARSA-SG)?
- ¿Cómo varía la velocidad de convergencia de Q-Learning vs SARSA en función de la densidad de recompensa?

El foco está en la **pregunta**, el **diseño**, el **análisis** y la **defensa oral** — no en que el agente llegue a una recompensa alta.

### Qué métricas tienen valor científico (y cuáles no)

El profesor distingue explícitamente entre métricas con fundamento y gráficas "estéticas".

**Métricas con respaldo teórico** (fuente: Sutton & Barto + transparencias):
- Evolución de la **recompensa media por episodio** (a corto y largo plazo)
- **Rendimiento promedio sobre múltiples ejecuciones independientes** (varias semillas)
- **Longitud media del episodio** hasta la convergencia — `f(t) = len(episodio_t)`
- **Tasa de éxito** (fracción de episodios completados correctamente)
- **Varianza del retorno** (estabilidad del aprendizaje)
- **Curvas de aprendizaje suavizadas** (media móvil para reducir ruido)
- **Análisis de estabilidad** frente a distintas semillas aleatorias
- **Error cuadrático medio** en estimaciones de valor (para evaluar calidad de Q)
- **Parámetros del algoritmo**: efecto de ε, α, γ, frecuencia de actualización de target network, tamaño del replay buffer...

**Gráficas que NO tienen valor metodológico** (el profesor lo dice explícitamente):
- Flechas de política sobre FrozenLake — "¿qué aporta en el análisis? ¿sirve en otros entornos?"
- Cualquier visualización que no responda a una pregunta experimental concreta
- Gráficas bonitas que no permiten interpretar convergencia ni comparar algoritmos

**La pregunta que hay que hacerse antes de incluir cualquier gráfica:**
> ¿A qué pregunta experimental responde esta gráfica? ¿Qué se puede concluir de ella?

Si no hay respuesta clara, la gráfica no va.

### Parámetros propios del algoritmo que también se estudian

No solo se estudia "si converge" — también se estudia **cómo afectan los parámetros**:
- Para Q-Learning: efecto de ε (exploración), α (tasa de aprendizaje), γ (descuento)
- Para DQN: tamaño del Replay Buffer, frecuencia de actualización de la Target Network, tamaño del batch, arquitectura de la red

Cada variación debe tener una **hipótesis previa** ("espero que un buffer más grande estabilice...") y un **análisis posterior** ("los resultados muestran que...").

---

## 1. Requisitos del Profesor (Entrega_Parte_RL.pdf)

### Qué entregar
- **PDF** de máximo 10 folios (sin portada, índice, bibliografía ni anexos)
- **Repositorio GitHub** con notebooks ejecutables en Colab sin errores
- **Fecha límite**: 15 de marzo de 2026
- **Exposición oral**: 2 y 5 de marzo de 2026

### Estructura esperada del repositorio
```
Entornos_Complejos/
├── src/agents/
│   ├── q_learning_agent.py        ← VACÍO, hay que implementar
│   └── dqn_agent.py               ← VACÍO, hay que implementar
├── QLearning_experiment.ipynb     ← VACÍO, hay que crear
├── DeepQLearning_experiment.ipynb ← VACÍO, hay que crear
└── CONTEXTO_PARTES_2_3.md         ← Este archivo
```

### Criterios de evaluación clave
- **Originalidad metodológica**: NO copiar experimentos de internet ni del repo del profesor
- **Rigor experimental**: justificar diseño, elegir métricas con fundamento científico
- **Reproducibilidad**: notebooks deben ejecutarse completos en Colab con "Ejecutar todas"
- **El informe PDF es autosuficiente**: no hay que mirar el código para entenderlo

### Métricas que pide (mínimo)
- Evolución de la **recompensa media acumulada** por episodio
- `f(t) = len(episodio_t)` — longitud de cada episodio (indicador de convergencia)
- Cualquier otra métrica con justificación teórica (tasa de éxito, curvas suavizadas, etc.)

### Entornos
- **Al menos 2** entornos de Gymnasium
- **Uno debe justificar** la necesidad de métodos aproximados (espacio de estados continuo)
- Evitar usar los mismos que el profesor: Taxi-v3, FrozenLake, LunarLander-v3 son los más "quemados"

---

## 2. Patrones del Código Existente

### Jerarquía de clases

```
GymnasiumAgent (ABC)              ← gymnasium_agent.py
    ├── SARSAAgent                ← sarsa_agent.py       (tabular, on-policy)
    ├── SARSASGAgent              ← sarsa_sg_agent.py    (red neuronal, on-policy)
    ├── MonteCarloOnPolicyAgent   ← monte_carlo_on_policy_agent.py
    └── MonteCarloOffPolicyAgent  ← monte_carlo_off_policy_agent.py
```

### Interfaz obligatoria de todo agente

```python
class MiAgente(GymnasiumAgent):

    def __init__(self, env, epsilon, discount_factor, decay, [params propios]):
        super().__init__(env, epsilon, discount_factor, decay)
        # super() crea: self.q_values, self.num_visited, estadísticas

    def get_action(self, obs):
        # La clase base ya implementa epsilon-greedy estándar
        # Solo sobreescribir si el agente usa red neuronal

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        # Aquí va el corazón del algoritmo

    def train(self, num_episodes):
        # Bucle de episodios con tqdm

    def stats(self):
        # Devuelve (list_stats, list_lengths)
        # Si hay red neuronal: (list_stats, list_lengths, list_losses)
```

### Bucle de entrenamiento estándar (del PDF, sección 5.2)

```python
for t in tqdm(range(num_episodes)):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        action = self.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        self.update(obs, action, next_obs, reward, terminated, truncated, info)
        obs = next_obs
        done = terminated or truncated
        episode_reward += reward
        episode_length += 1

    if self.decay:
        self.epsilon = min(1.0, 1000.0 / (self._episode_count + 1))

    self.update_stats(episode_reward, episode_length)
```

### Decaimiento de epsilon
- **Tabular**: `epsilon = min(1.0, 1000.0 / (t + 1))` — decae rápido (100k episodios)
- **Red neuronal**: `epsilon = max(0.01, epsilon * 0.9977)` — decae exponencial (llega a 0.01 en episodio ~2000)

### Red neuronal del profesor (sección 5.3, plantilla DQN_Network)

```python
class DQN_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )
        # Inicialización Kaiming para ReLU
        for layer in self.FC:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        return self.FC(x)
```

> El `sarsa_sg_agent.py` tiene una versión mejorada con `hidden_size` y `num_hidden_layers` configurables.

---

## 3. Algoritmos a Implementar

### Parte 2: Q-Learning Tabular (`q_learning_agent.py`)

**Diferencia con SARSA**: es off-policy. El target usa el `max` sobre todas las acciones, no la acción que realmente se va a tomar.

**Fórmula de actualización**:
```
Q(S,A) ← Q(S,A) + α * [R + γ * max_a Q(S',a) - Q(S,A)]
```

**Consecuencias en el código respecto a SARSA**:
1. En `update()`: NO necesita `next_action` como parámetro → usa `np.max(self.q_values[next_obs])`
2. En `train()`: el bucle es más simple (no hay que pre-seleccionar `next_action` antes de update)
3. Hereda de `GymnasiumAgent` directamente (entorno discreto → tiene `observation_space.n`)

**Parámetros adicionales al constructor**: `alpha` (igual que SARSAAgent)

### Parte 3: Deep Q-Learning (`dqn_agent.py`)

**Diferencias con SARSA semi-gradiente** (que ya existe):

| Característica | SARSA-SG | DQN |
|---|---|---|
| Replay Buffer | No | Sí — guarda (s,a,r,s',done) y sampea batches |
| Target Network | No (una sola red) | Sí — segunda red congelada, actualización periódica |
| Política | On-policy | Off-policy |
| Target del update | `Q(s', a')` con la acción tomada | `max_a Q_target(s', a)` |
| Update | Paso a paso | Por minibatch (ej. 64 transiciones) |

**Algoritmo (del paper de Atari, Algorithm 1)**:
```
Inicializar:
  - red principal Q con pesos w
  - red objetivo Q_target con pesos w_target = w
  - replay buffer D (capacidad N)

Para cada episodio:
  Para cada paso t:
    1. Seleccionar acción con epsilon-greedy usando Q
    2. Ejecutar acción, observar (r, s', done)
    3. Guardar (s, a, r, s', done) en D
    4. Si |D| >= batch_size:
       - Sampear minibatch de D
       - Calcular target: y = r  si done
                          y = r + γ * max_a Q_target(s', a)  si not done
       - Calcular loss = MSE(Q(s,a), y)
       - Actualizar w con descenso de gradiente
    5. Cada C pasos: actualizar w_target = w  (o soft update)
```

**No hereda** `super().__init__()` directamente (entorno continuo, no tiene `observation_space.n`).
Inicializa estadísticas manualmente igual que `SARSASGAgent`.

---

## 4. Utilidades de Plotting Disponibles

Archivo: `src/plotting/plotting.py`

| Función | Uso |
|---|---|
| `plot(list_stats, title, ylabel)` | Recompensa media acumulada |
| `plot_lengths(list_lengths, window)` | Longitud de episodios con media móvil |
| `plot_losses(list_losses, title)` | Loss de entrenamiento (para redes) |
| `plot_comparison(list_of_stats, labels, ...)` | Comparar varios agentes |
| `plot_lengths_comparison(...)` | Comparar longitudes de varios agentes |
| `show_greedy_episode(env, Q, ...)` | Ejecutar política greedy final (tabular) |
| `show_greedy_episode_qnet(env, agent, ...)` | Ejecutar política greedy final (red neuronal) |
| `frames_to_gif(frames, filename)` | Generar GIF del episodio |

---

## 5. Plan de Implementación (Paso a Paso)

### Paso 1 — `q_learning_agent.py`
- Copiar estructura de `sarsa_agent.py`
- Cambiar `update()`: eliminar `next_action`, usar `np.max(self.q_values[next_obs])`
- Simplificar `train()`: no pre-seleccionar `next_action`

### Paso 2 — `QLearning_experiment.ipynb`
- Elegir entorno tabular (discreto)
- Comparar Q-Learning vs SARSA en el mismo entorno
- Graficar recompensa y longitud de episodios
- Analizar diferencias on-policy vs off-policy

### Paso 3 — `dqn_agent.py`
- Definir clase `ReplayBuffer` (deque de capacidad máxima, método `sample`)
- Definir red neuronal (reutilizar `DQN_Network` de sarsa_sg o adaptar)
- Implementar `DQNAgent` con: red principal, red target, optimizer, buffer
- `update()` trabaja con batches, no con transiciones individuales
- `train()` incluye lógica de: cuándo actualizar target network, warm-up del buffer

### Paso 4 — `DeepQLearning_experiment.ipynb`
- Elegir entorno continuo que justifique DQN
- Demostrar por qué Q-Learning tabular no funciona ahí
- Entrenar DQN y mostrar convergencia
- Comparar con SARSA semi-gradiente si aplica

---

## 6. Entornos Elegidos y Justificación Experimental

### Decisión final
| Parte | Entorno | Algoritmos que compara |
|---|---|---|
| Q-Learning (tabular) | `Taxi-v3` | Q-Learning vs SARSA vs MC On-Policy vs MC Off-Policy (ya hechos) |
| DQN (aproximado) | `LunarLander-v3` | DQN vs SARSA semi-gradiente (ya hecho) |

### Por qué esto es válido académicamente

Usar el mismo entorno que experimentos anteriores **no es falta de originalidad** si la comparación tiene hipótesis propias. El entorno es la variable controlada — lo que varía es el algoritmo. Esto es diseño experimental correcto.

### Hipótesis que deben articularse en el informe

**Para Q-Learning en Taxi-v3:**
> *"La comparación cubre el espacio {on/off-policy} × {TD/MC}: Q-Learning (off-policy, TD), SARSA (on-policy, TD), MC Off-Policy (off-policy, episódico), MC On-Policy (on-policy, episódico). Esperamos que los métodos TD converjan más rápido que los MC por usar bootstrapping. Entre los off-policy, Q-Learning debería superar a MC Off-Policy en velocidad de convergencia pero con mayor varianza durante el entrenamiento. SARSA, al seguir la política de comportamiento, debería ser más conservador y estable."*

Métricas que lo verifican:
- Recompensa media por episodio (¿quién llega antes a valores positivos?)
- Varianza del retorno episodio a episodio (¿quién es más estable?)
- Longitud media del episodio hasta convergencia

**Para DQN en LunarLander-v3:**
> *"El Replay Buffer rompe la correlación entre transiciones consecutivas y la Target Network estabiliza el objetivo de aprendizaje. Esperamos que DQN sea más estable que SARSA semi-gradiente (menos oscilaciones en la curva de recompensa) aunque posiblemente más lento en los primeros episodios por necesitar llenar el buffer."*

Métricas que lo verifican:
- Comparación de curvas de recompensa suavizadas (varianza visible)
- Curva de loss durante entrenamiento (DQN vs SARSA-SG)
- Recompensa media final tras N episodios

---

## 7. Lecturas de Referencia

| Documento | Para qué |
|---|---|
| `Entrega_Parte_RL.pdf` §4.2, §5.2, §5.3 | Normas, esquema Gymnasium, plantilla PyTorch |
| `Transpa-DiferenciasTemporales.pdf` | Pseudocódigo Q-Learning, teoría off-policy |
| `Transpa-ApproxControl.pdf` | Arquitectura DQN, Replay Buffer, Target Network |
| `Playing_Atari_with_Deep_Reinforcement_Learning.pdf` | Algorithm 1: bucle DQN completo |
| `Deep_Reinforcement_Learning_with_Double_Q-learning.pdf` | (Opcional) Double DQN para evitar sobreestimación |

---

## 8. Guía Detallada de Lecturas

Para una explicación completa de **qué leer exactamente en cada documento, en qué orden y qué extraer**,
consulta el documento específico:

**[LECTURAS_REFERENCIA.md](LECTURAS_REFERENCIA.md)**

Ese archivo contiene:
- Tabla sección a sección de `Entrega_Parte_RL.pdf` (normas y plantillas de código)
- La fórmula clave de Q-Learning destacada y comparada con SARSA
- Los dos componentes de DQN (Replay Buffer y Target Network) explicados en detalle
- El pseudocódigo completo del Algorithm 1 del paper de Atari
- El mapa de dónde están físicamente todos los documentos en el repositorio
