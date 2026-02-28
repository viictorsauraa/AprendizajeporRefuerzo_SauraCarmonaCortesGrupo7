"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de resultados de entrenamiento de agentes en entornos de Gymnasium.
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio


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

def plot_comparison(list_of_stats, labels, title='Comparativa de Algoritmos', ylabel='Recompensa Media'):
    """
    Grafica múltiples series de datos para comparar su evolución.
    :param list_of_stats: Lista que contiene las listas de estadísticas (ej. [stats_sarsa, stats_mc])
    :param labels: Lista con los nombres para la leyenda (ej. ['SARSA', 'Monte Carlo'])
    """
    plt.figure(figsize=(10, 5))
    
    for stats, label in zip(list_of_stats, labels):
        indices = list(range(len(stats)))
        plt.plot(indices, stats, label=label)
    
    plt.title(title)
    plt.xlabel('Episodio')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend() # Muestra la leyenda con los nombres de los algoritmos
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

def plot_lengths_comparison(list_of_lengths, labels, window=500):
    """
    Compara la longitud de los episodios de varios agentes con sus 
    respectivas curvas de tendencia (media móvil).
    
    :param list_of_lengths: Lista de listas con las longitudes (ej. [len_sarsa, len_mc])
    :param labels: Etiquetas para la leyenda (ej. ['SARSA', 'Monte Carlo'])
    :param window: Ventana para el cálculo de la media móvil.
    """
    plt.figure(figsize=(10, 5))
    
    # Definir una paleta de colores para diferenciar las tendencias
    colors = ['crimson', 'teal', 'darkorange', 'purple']
    
    for i, (lengths, label) in enumerate(zip(list_of_lengths, labels)):
        n = len(lengths)
        current_window = min(window, max(1, n // 10))
        color = colors[i % len(colors)]
        
        # Graficar los datos crudos con mucha transparencia (alpha bajo) para no saturar
        plt.plot(lengths, alpha=0.15, color=color, linewidth=0.5)
        
        # Calcular y graficar la media móvil (Tendencia)
        if n >= current_window and current_window > 1:
            moving_avg = np.convolve(lengths, np.ones(current_window) / current_window, mode='valid')
            plt.plot(range(current_window - 1, n), moving_avg,
                     color=color, linewidth=2, label=f'Tendencia {label}')
    
    plt.title('Comparativa: Eficiencia en pasos por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Número de pasos')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
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

def show_greedy_episode_qnet(env, agent, max_steps=500):
    """
    Ejecuta un episodio usando la política greedy y captura los fotogramas.

    Parámetros:
      - env: Entorno Gymnasium configurado con render_mode='rgb_array'.
      - agent: Agente entrenado con una red neuronal Q.
      - max_steps (int): Número máximo de pasos a ejecutar en el episodio.

    Retorna:
      - list: Lista de fotogramas (imágenes) capturados durante el episodio.
    """
    frames = []  # Lista para almacenar cada fotograma.
    agent.epsilon = 0  # Aseguramos que el agente actúe de forma completamente greedy.

    q_network = agent.network  # Acceder a la red neuronal del agente

    # Reiniciar el entorno y obtener el estado inicial.
    state, _ = env.reset()
    done = False  # Indicador de finalización del episodio.

    # Ejecutar el episodio hasta max_steps o hasta que el entorno indique que ha terminado.
    for _ in range(max_steps):
        # Capturar el fotograma actual del entorno.
        frame = env.render()
        frames.append(frame)

        # Seleccionar la acción óptima utilizando la función greedy.
        action = agent.get_action(state)  # Esto usará la política greedy debido a epsilon=0

        # Ejecutar la acción en el entorno y obtener el siguiente estado y otros datos.
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state  # Actualizar el estado.

        # Si el episodio ha terminado o se ha truncado, capturar el fotograma final y salir.
        if done or truncated:
            frames.append(env.render())
            break

    return frames


def frames_to_gif(frames, filename="lunar_landing_sarsa_sg.gif"):
    """
    Crea un archivo GIF a partir de una lista de fotogramas.

    Parámetros:
      - frames (list): Lista de fotogramas (imágenes) capturados del entorno.
      - filename (str): Nombre del archivo GIF resultante.

    Retorna:
      - str: Nombre del archivo GIF creado.
    """
    # Abrir un escritor de GIF con imageio.
    with imageio.get_writer(filename, mode='I') as writer:
        # Agregar cada fotograma al GIF.
        for frame in frames:
            writer.append_data(frame)
    return filename
     

def plot_policy_taxi(env, Q, passenger_loc, destination_idx):
    """
    Dibuja un mapa de 5x5 con la mejor acción para cada celda
    dado un estado del pasajero y un destino, y muestra la 
    representación visual inicial del entorno.
    """
    # 1. Configurar el entorno en el estado específico para obtener el renderizado visual
    # El taxi se coloca por defecto en (0,0) para la visualización inicial del mapa
    initial_state = env.unwrapped.encode(0, 0, passenger_loc, destination_idx)
    env.unwrapped.s = initial_state
    
    # 2. Mostrar el estado inicial como imagen
    frame = env.render()
    print(f"\nEstado Inicial del Entorno (Pasajero: {passenger_loc}, Destino: {destination_idx}):")
    if isinstance(frame, np.ndarray):
        plt.figure(figsize=(3, 3))
        plt.imshow(frame)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(frame)

    # 3. Dibujar el Mapa de Política con flechas
    # Mapeo de acciones a símbolos: 0: Sur, 1: Norte, 2: Este, 3: Oeste, 4: Pickup, 5: Dropoff
    symbols = {0: " ↓ ", 1: " ↑ ", 2: " → ", 3: " ← ", 4: " P ", 5: " D "}
    
    print(f"Mapa de Política (Mejor acción por celda):")
    print("-" * 25)
    
    for row in range(5):
        row_str = "|"
        for col in range(5):
            # Codificamos el estado para cada posición posible del taxi
            state = env.unwrapped.encode(row, col, passenger_loc, destination_idx)
            
            # Buscamos la mejor acción según la tabla Q
            best_action = np.argmax(Q[state])
            row_str += symbols[best_action] + "|"
        print(row_str)
    print("-" * 25)


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

def show_greedy_episode_img(env, Q, max_steps=200, seed=None, title="Episodio greedy"):
    """
    Ejecuta un episodio con la política greedy (argmax Q) y muestra el estado
    inicial y el estado final como imágenes (requiere render_mode='rgb_array').
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
    status = 'Éxito' if (done and total_reward > -steps) else ('Truncado' if not done else 'Terminado')
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(initial_frame)
    axes[0].set_title('Estado inicial')
    axes[0].axis('off')
    axes[1].imshow(final_frame)
    axes[1].set_title(f'Estado final ({status})')
    axes[1].axis('off')
    fig.suptitle(f"{title}  |  Recompensa: {total_reward:.0f}  |  Pasos: {steps}")
    plt.tight_layout()
    plt.show()


def plot_losses(list_losses, window=1000, title='Evolución de la pérdida (Loss)'):
    """Gráfica de la evolución de la pérdida con media móvil y escala logarítmica."""
    n = len(list_losses)
    window = min(window, max(1, n // 50))

    plt.figure(figsize=(10, 4))

    plt.plot(list_losses, alpha=0.15, color='orange', label='Pérdida (paso a paso)')

    if n >= window and window > 1:
        moving_avg = np.convolve(list_losses, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, n), moving_avg,
                 color='darkred', linewidth=1.5,
                 label=f'Tendencia (media móvil, ventana={window})')

    # log permite ver qué pasa en los valores pequeños sin que los picos grandes lo tapen todo.
    plt.yscale('log')

    plt.title(title)
    plt.xlabel('Paso de entrenamiento (Step)')
    plt.ylabel('Pérdida (Loss) - Escala Log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()