# Pasos Siguientes — Entornos Complejos

Estado a fecha 2026-02-22. La parte A y el Paso 1 (MC off-policy) están completados.

---

## ✅ Ya hecho (Parte A)
- `MonteCarloTodasLasVisitas.ipynb` con agente MC on-policy esquema Gymnasium 5.2.
- Retornos corregidos (hacia atrás), `plot_lengths`, `plot_scenario`.
- Verificado teóricamente frente a Algoritmo 3 de `Transpa-MonteCarlo.pdf`.

---

## Parte B — Métodos Tabulares

### ✅ Paso 1 — Monte Carlo off-policy `MonteCarloOffPolicy.ipynb`

**COMPLETADO.** Fichero: `src/MonteCarloOffPolicy.ipynb`. Ejecutado en `tests/MonteCarloOffPolicy.ipynb`.

Algoritmo implementado: **Algoritmo 6** del PDF (off-policy control con IS ponderado). Se eligió el 6 porque:
- Alg. 4: predicción de V(s) para π fija — no hace control.
- Alg. 5: predicción de Q(s,a) para π fija — no mejora la política.
- Alg. 6: control — mejora π iterativamente. Único aplicable para encontrar política óptima.

Resultados (50 000 episodios, semilla fija):
| Entorno | Parámetros | Tasa éxito |
|---------|-----------|------------|
| 4×4 | ε=0.4, decay=True | **84.1%** |
| 8×8 | ε=0.4, decay=True, q_init=0.01 | **84.2%** |

Hallazgos clave:
- **decay obligatorio en ambos** entornos: con ε constante, P(backward completo k pasos) = 0.7^k → 12% para k=6.
- **q_init necesario en 8×8**: Q=0 → argmax siempre devuelve LEFT (acción 0) → agente atrapado en esquina. q_init=0.01 rompe el sesgo.
- **Por qué q_init sobrevive**: el break del backward loop protege la mayoría de Q[s,a] de ser actualizados en episodios fallidos. La tabla Q del 8×8 muestra tres poblaciones: ~1.0 (trayectoria greedy), ~0.001–0.009 (q_init sin actualizar), 0.0 (agujeros).
- **Validez**: Alg. 6 dice "Q(s,a) ∈ ℝ, arbitrarily" → inicialización aleatoria explícitamente válida.

---

### Paso 2 — SARSA (TD on-policy) `SARSA.ipynb`

Algoritmo: **SARSA** — actualización TD en cada paso, no al final del episodio.

```
Q(S,A) ← Q(S,A) + α·[R + γ·Q(S',A') − Q(S,A)]
```

Aspectos clave:
- Actualización paso a paso (no hay que esperar al fin del episodio).
- Política: ε-greedy sobre Q.
- `update` recibe (s, a, s', r, terminado) y actualiza inmediatamente.
- Parámetros: α (tasa de aprendizaje), γ, ε.
- Probar en FrozenLake 4×4 y 8×8. Comparar convergencia con MC.

Estructura del agente:
```python
class SARSAAgent:
    def __init__(self, env, alpha=0.1, epsilon=0.1, discount_factor=1.0): ...
    def get_action(self, state): ...
    def update(self, obs, action, next_obs, reward, terminated, truncated, info): ...
    def stats(self): ...
```

---

### Paso 3 — Q-Learning (TD off-policy) `QLearning.ipynb`

Algoritmo: **Q-Learning** — como SARSA pero la actualización usa el máximo futuro.

```
Q(S,A) ← Q(S,A) + α·[R + γ·max_a Q(S',a) − Q(S,A)]
```

Aspectos clave:
- Off-policy: sigue ε-greedy para explorar, pero actualiza con greedy.
- Generalmente más agresivo en la actualización que SARSA.
- Comparar con SARSA: velocidad de convergencia, estabilidad.
- Probar en FrozenLake y en un segundo entorno (ver Paso 5).

---

### Estudio comparativo Tabular

Notebook dedicado o sección dentro de los anteriores. Ideas para hipótesis propias:
- ¿SARSA vs Q-Learning en entornos con "acantilados" (cliffwalking)?
- ¿MC on-policy vs off-policy: sesgo-varianza con distintos ε?
- ¿Cómo afecta γ < 1 a la longitud de episodios aprendidos?
- ¿Qué pasa con distintos α en SARSA? ¿Hay α óptimo para FrozenLake 8×8?

---

## Parte B — Control con Aproximaciones

### Paso 4 — SARSA semi-gradiente `SARSASemiGradiente.ipynb`

Para entornos con **espacio de estados continuo** (requiere aproximación).

Algoritmo: SARSA con función de valor aproximada `Q̂(s,a,w)`.

```
w ← w + α·[R + γ·Q̂(S',A',w) − Q̂(S,A,w)]·∇_w Q̂(S,A,w)
```

Opciones de aproximación:
- **Lineal**: features manuales (tile coding, RBF, polinomios) → `Q̂ = w^T · φ(s,a)`
- Entorno sugerido: **CartPole-v1** (espacio continuo, 4 variables de estado).

Estructura del agente:
```python
class SARSASemiGradientAgent:
    def __init__(self, env, alpha=0.001, epsilon=0.1, discount_factor=1.0): ...
    def get_action(self, state): ...
    def update(self, obs, action, next_obs, reward, terminated, truncated, info): ...
    def stats(self): ...
```

---

### Paso 5 — Deep Q-Learning (DQN) `DQN.ipynb`

Para entornos con **espacio de estados muy grande o continuo**.

Algoritmo: Q-Learning con red neuronal como función Q̂(s,a).

Componentes clave:
- Red neuronal `Q̂(s,a,θ)` (PyTorch, según sección 5.3 del PDF).
- **Experience Replay**: buffer de transiciones (s,a,r,s',done).
- **Target Network**: red de target fija que se actualiza periódicamente.
- Loss: `L = E[(R + γ·max_a Q̂(s',a,θ⁻) − Q̂(s,a,θ))²]`

```python
class DQNAgent:
    def __init__(self, env, lr=1e-3, epsilon=0.1, gamma=0.99,
                 buffer_size=10000, batch_size=64, target_update=100): ...
    def get_action(self, state): ...
    def update(self, obs, action, next_obs, reward, terminated, truncated, info): ...
    def stats(self): ...
```

Entorno sugerido: **CartPole-v1** o **LunarLander-v2**.

---

## Selección de entornos

| Entorno | Tipo espacio estados | Método adecuado | Justificación |
|---------|---------------------|-----------------|---------------|
| FrozenLake 4×4 / 8×8 | Discreto finito | MC, SARSA, Q-Learning | Tabular directo |
| CartPole-v1 | Continuo (4 vars) | SARSA semi-grad, DQN | Requiere aproximación |
| LunarLander-v2 (opcional) | Continuo (8 vars) | DQN | Más complejo, necesita NN |

> El contraste entre FrozenLake (tabular ✓) y CartPole (aproximación obligatoria) pone de manifiesto directamente cuándo son necesarios los métodos aproximados.

---

## Orden de implementación recomendado

```
[HECHO] A) MonteCarloTodasLasVisitas.ipynb
   ↓
[HECHO] 1) MonteCarloOffPolicy.ipynb
   ↓
[PASO 2] SARSA.ipynb  ←→  [PASO 3] QLearning.ipynb
   ↓ (estudio comparativo tabular)
[PASO 4] SARSASemiGradiente.ipynb  (CartPole)
   ↓
[PASO 5] DQN.ipynb  (CartPole / LunarLander)
   ↓ (estudio comparativo aproximaciones)
```

---

## Recordatorio de restricciones

- Semilla fija en todos los notebooks para reproducibilidad.
- Pocas gráficas, bien elegidas (no una por variante de parámetro).
- Estudio propio: hipótesis originales, no réplica de tutoriales online.
- Todo reproducible en Colab con un solo click.
- Usar el esquema de agente de la sección 5.2 en **todos** los notebooks.
