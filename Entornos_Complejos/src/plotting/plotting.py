"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de resultados de entrenamiento de agentes en entornos de Gymnasium.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot(list_stats):
  indices = list(range(len(list_stats)))
  plt.figure(figsize=(6, 3))
  plt.plot(indices, list_stats)
  plt.title('Proporción de recompensas')
  plt.xlabel('Episodio')
  plt.ylabel('Proporción')
  plt.grid(True)
  plt.show()

def plot_lengths(list_lengths, window=500):
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