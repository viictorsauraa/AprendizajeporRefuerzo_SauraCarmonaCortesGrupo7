"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de resultados de entrenamiento de agentes en entornos de Gymnasium.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot(list_stats, title='Proporción de recompensas', ylabel='Proporción'):
    """Evolución de la métrica media acumulada por episodio."""
    indices = list(range(len(list_stats)))
    plt.figure(figsize=(8, 3))
    plt.plot(indices, list_stats)
    plt.title(title)
    plt.xlabel('Episodio')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_lengths(list_lengths, window=500):
    """Longitud de los episodios con curva de tendencia (media móvil)."""
    n = len(list_lengths)
    window = min(window, max(1, n // 10))
    plt.figure(figsize=(10, 4))
    plt.plot(list_lengths, alpha=0.25, color='steelblue',
             linewidth=0.8, label='Longitud del episodio')
    if n >= window and window > 1:
        moving_avg = np.convolve(list_lengths, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, n), moving_avg,
                 color='crimson', linewidth=2,
                 label=f'Tendencia (media móvil, ventana={window})')
    plt.title('Longitud de los episodios')
    plt.xlabel('Episodio')
    plt.ylabel('Número de pasos')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_scenario(env, Q, title='Política óptima aprendida'):
    """Grid FrozenLake con flechas de la política óptima."""
    desc = env.unwrapped.desc
    nrows, ncols = desc.shape
    cell_colors = {b'S': '#90EE90', b'F': '#ADD8E6', b'H': '#555555', b'G': '#FFD700'}
    arrows = {0: chr(8592), 1: chr(8595), 2: chr(8594), 3: chr(8593)}
    fig, ax = plt.subplots(figsize=(ncols * 1.2, nrows * 1.2))
    for row in range(nrows):
        for col in range(ncols):
            cell = desc[row, col]
            color = cell_colors.get(cell, 'white')
            y = nrows - 1 - row
            ax.add_patch(plt.Rectangle([col, y], 1, 1, color=color, ec='black', lw=0.8))
            state = row * ncols + col
            cx, cy = col + 0.5, y + 0.5
            if cell == b'H':
                ax.text(cx, cy, 'H', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
            elif cell == b'G':
                ax.text(cx, cy, 'G', ha='center', va='center', fontsize=12, fontweight='bold', color='black')
            else:
                best_action = np.argmax(Q[state])
                ax.text(cx, cy, arrows[best_action], ha='center', va='center', fontsize=16)
                if cell == b'S':
                    ax.text(col + 0.12, y + 0.88, 'S', ha='center', va='center',
                            fontsize=7, color='darkgreen', fontweight='bold')
    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    plt.tight_layout(); plt.show()


def show_greedy_episode(env, Q, max_steps=200, seed=None, title="Episodio greedy"):
    """
    Ejecuta un episodio con la política greedy (argmax Q) y muestra el estado
    inicial y el estado final en formato ANSI (requiere render_mode='ansi').
    """
    state, info = env.reset(seed=seed) if seed is not None else env.reset()
    initial_frame = env.render()
    done = False
    total_reward = 0
    steps = 0
    while not done and steps < max_steps:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    final_frame = env.render()
    if done and total_reward > -steps:
        status = 'Éxito'
    elif not done:
        status = 'Truncado (no convergió)'
    else:
        status = 'Terminado'
    print(f"{title}  |  Recompensa: {total_reward:.0f}  |  Pasos: {steps}  |  {status}")
    print("\nEstado inicial:")
    print(initial_frame)
    print("Estado final:")
    print(final_frame)


def print_q_summary(env, Q, title="Resumen tabla Q"):
    """
    Resumen estadístico de la tabla Q aprendida.
    Si el entorno es Taxi-v3 (tiene env.unwrapped.decode), muestra una muestra
    de estados decodificados; en caso contrario, solo estadísticas globales.
    """
    non_zero = np.count_nonzero(Q)
    total = Q.size
    print(f"--- {title} ---")
    print(f"  Entradas no nulas : {non_zero}/{total} ({100 * non_zero / total:.1f}%)")
    print(f"  Max Q             : {Q.max():.3f}")
    if non_zero > 0:
        print(f"  Min Q (no nulo)   : {Q[Q != 0].min():.3f}")
        print(f"  Q media (no nulo) : {Q[Q != 0].mean():.3f}")

    # Decodificación específica de Taxi-v3
    try:
        decode = env.unwrapped.decode
        locs = ['R', 'G', 'Y', 'B', 'en taxi']
        action_labels = ['S', 'N', 'E', 'W', 'PU', 'DO']
        sample_states = [int(s) for s in np.linspace(0, Q.shape[0] - 1, 5, dtype=int)]
        print("\n  Muestra de estados:")
        for state in sample_states:
            r, c, pl, di = decode(state)
            q = Q[state]
            best = int(np.argmax(q))
            print(f"  [{state:3d}] taxi=({r},{c}) pas={locs[pl]:7s} dst={locs[di]:2s} "
                  f"-> mejor={action_labels[best]} Q={q.round(2)}")
    except (AttributeError, TypeError):
        pass
