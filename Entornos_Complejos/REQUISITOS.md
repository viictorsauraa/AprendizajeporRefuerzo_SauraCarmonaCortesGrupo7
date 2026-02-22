# Requisitos — Grupo 2: Aprendizaje en Entornos Complejos
*Extraído de `Entrega_Parte_RL.pdf`, páginas 14–16*

---

## Contexto general
El objetivo es estudiar algoritmos de RL en entornos complejos usando **Gymnasium** como framework. Las tareas se dividen en dos bloques: conocer el entorno y estudiar algoritmos básicos.

---

## A) Conocer el entorno Gymnasium

> *"Para entender su funcionamiento debe de entender: el último apartado (sección 5.2), la documentación básica de Gymnasium, y su aplicación en el notebook MonteCarloTodasLasVisitas.ipynb"*

### ¿Qué se pide?

**Modificar el notebook `MonteCarloTodasLasVisitas.ipynb`** para que haga lo siguiente:

1. **Construir un agente** de acuerdo a las indicaciones de Gymnasium (sección 5.2):
   - Esquema: `__init__`, `get_action(state)`, `update(obs, action, next_obs, reward, terminated, truncated, info)`
   - El diseño del agente se usará en el resto de la práctica.

2. **Corregir el algoritmo Monte Carlo**: la versión original calcula los retornos *hacia adelante* (bug intencionado). Debe calcularse *hacia atrás* (G = r + γ·G).

3. **Añadir una gráfica adicional**: `f(t) = len(episodio_t)`.
   - Justificar por qué esta gráfica también es un buen indicador de aprendizaje.

### Estado actual ✅ COMPLETADO
- Agente `MonteCarloOnPolicyAgent` implementado con esquema Gymnasium 5.2.
- Retornos calculados correctamente hacia atrás.
- `plot_lengths` añadida con media móvil y justificación en 3 fases.
- `plot_scenario` añadida (visualización del grid con política óptima).
- Verificación teórica frente al Algoritmo 3 del PDF `Transpa-MonteCarlo.pdf`: ✅ correcto.

---

## B) Estudio de algunos algoritmos básicos

> *"Desarrolle estudios comparativos entre las distintas técnicas vistas en clase."*

### Parte 2 — Métodos Tabulares

| Algoritmo | Tipo | Estado |
|-----------|------|--------|
| Monte Carlo on-policy | Tabular | ✅ hecho (MonteCarloTodasLasVisitas.ipynb) |
| Monte Carlo off-policy | Tabular | ✅ hecho (MonteCarloOffPolicy.ipynb) |
| SARSA | TD tabular | ⬜ pendiente |
| Q-Learning | TD tabular | ⬜ pendiente |

### Parte 3 — Control con Aproximaciones

| Algoritmo | Tipo | Estado |
|-----------|------|--------|
| SARSA semi-gradiente | Aproximación lineal | ⬜ pendiente |
| Deep Q-Learning (DQN) | Aproximación con red neuronal | ⬜ pendiente |

> Nota: **No se pide ningún algoritmo Actor-Crítico**.

### Entornos requeridos
- Al menos **dos entornos de Gymnasium**.
- Uno de ellos debe **poner de manifiesto la necesidad de métodos aproximados** (e.g., espacio de estados continuo).
- Entornos válidos: Classic Control, Box2D, Toy Text, MuJoCo, Atari, o externos (Bluesky, Flappy-bird, Tetris, Sokoban, Simplegrid...).
- **Requisito crítico**: ejecución inmediata en Colab sin instalación manual de software.

### Restricciones importantes

**Sobre las gráficas:**
- Pocas gráficas, bien elegidas. Sería un error hacer 10 gráficas para 10 algoritmos.
- Se pueden modificar los parámetros de las funciones de dibujo obligatorias.

**Sobre la reproducibilidad:**
- Usar siempre la **misma semilla** para que las gráficas sean reproducibles.

**Sobre la documentación:**
- Documentación escrita limitada, pero el contenido del GitHub no lo es.
- Los notebooks pueden contener toda la experimentación que se considere.

**Sobre la originalidad (⚠️ IMPORTANTE):**
- **No reproducir experimentos de internet.** Riesgo de plagio aunque no sea copia literal.
- No usar los mismos entornos, parámetros, patrones visuales, conclusiones o estudios futuros que puedan encontrarse fácilmente en la red.
- **Buscad vuestro propio estudio experimental, con criterio y justificación.**
- Los estudios forman parte de la evaluación: formular hipótesis, criticar, experimentar, juzgar, comprobar coherencia.

---

## Esquema general del agente (sección 5.2)

```python
class Agent:
    def __init__(self, env: gym.Env, hiperparámetros):
        """Inicializa todo lo necesario para el aprendizaje"""

    def get_action(self, state) -> Any:
        """Indica qué acción realizar. Responde a la política del agente."""

    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """Aplica el algoritmo de aprendizaje con la muestra (s, a, s', r)."""

    def stats(self):
        """Retorna resultados estadísticos y de evolución."""
```

Bucle de entrenamiento episódico estándar:
```python
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, next_obs, reward, terminated, truncated, info)
        done = terminated or truncated
        obs = next_obs
stats = agent.stats()
```
