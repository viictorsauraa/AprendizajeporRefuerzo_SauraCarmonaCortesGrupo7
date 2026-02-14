# Notas de estudio: k-Armed Bandit con Epsilon-Greedy

## Análisis de gráficas: cuáles son las más relevantes

### Gráficas pedidas

| Gráfica | Función | Obligatoria |
|---|---|---|
| Recompensa promedio | `plot_average_rewards` | Sí (ya estaba) |
| % Selección brazo óptimo | `plot_optimal_selections` | Sí |
| Regret acumulado | `plot_regret` | Sí |
| Estadísticas por brazo | `plot_arm_statistics` | Opcional |

---

### Conclusión: las más importantes son regret acumulado y % selecciones óptimas

#### 1. Regret acumulado — la más importante

El regret es la **métrica formal del problema bandido**: coste total de no haber elegido siempre
el brazo óptimo. Es la que más información aporta sobre epsilon-greedy porque revela su
**limitación teórica estructural**:

- Para ε > 0: exploración con probabilidad fija en *cada* paso → **regret lineal O(ε·T)**.
  La curva no se aplana nunca, independientemente de cuánto haya aprendido el algoritmo.
- Para ε = 0: puede tener regret aún mayor si se engancha a un brazo subóptimo al inicio,
  sin mecanismo de corrección.
- Comparar con la cota C·ln(T) (incluida en el notebook) evidencia que epsilon-greedy
  **no alcanza regret logarítmico**, a diferencia de UCB o Thompson Sampling.

Esta conclusión solo es visible en la gráfica de regret, no en las demás.

#### 2. % Selección del brazo óptimo — la segunda más importante

Traduce el balance exploración-explotación al espacio de decisiones:

- ε = 0.1 → converge rápido pero se estabiliza en ~90% (el 10% siempre explora).
- ε = 0.01 → converge más lento pero se acerca más al 100%.
- ε = 0 → puede quedar atrapado en un brazo subóptimo indefinidamente.

Muestra el **dilema central del bandido** y por qué no hay un ε universalmente óptimo.

#### Por qué las otras dos son secundarias

- **Recompensa promedio**: mezcla la calidad del brazo con el aprendizaje; no separa
  exploración de explotación.
- **Estadísticas por brazo**: diagnóstico de qué ocurrió en un experimento concreto,
  no comparación entre políticas.

**Conclusión final**: regret acumulado + % selecciones óptimas forman el par mínimo suficiente.
La primera responde *cuánto costó* la estrategia de exploración; la segunda, *con qué frecuencia
la política convergió a la decisión correcta*. Juntas permiten argumentar tanto la ventaja de
explorar (frente a ε=0) como la limitación de explorar con tasa fija (frente a algoritmos
adaptativos como UCB).

---

## Cambios realizados en el código

### `arms/armbernoulli.py`
- **Corrección**: `ArmBernoulli` ahora hereda de `ArmBinomial` en lugar de `Arm`.
  Refleja que Bernoulli es un caso particular de Binomial con n=1.
- Constructor: `super().__init__(n=1, p=p)` — toda la lógica de `pull()` y
  `can_approximate_normal()` se hereda sin duplicación.
- Jerarquía resultante: `Arm → ArmBinomial → ArmBernoulli`

### `plotting/plotting.py`
- **Corrección**: `plot_optimal_selections` — `ylim([-2, 100])` sustituido por
  `ylim([0, 105])`. Un porcentaje no puede ser negativo.
- **Añadido**: `plot_arm_statistics` — leyenda global de colores con `matplotlib.patches`
  (verde=brazo óptimo, azul=brazo subóptimo, gris=no seleccionado). Exigido implícitamente
  por el enunciado ("añade lo que consideres necesario para clarificar la gráfica").
