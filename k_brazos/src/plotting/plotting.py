"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from algorithms import Algorithm, EpsilonGreedy, EpsilonDecaimiento, Softmax, UCB1


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, Softmax):
        label += f" (tau={algo.tau})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    elif isinstance(algo, EpsilonDecaimiento):
        label += f" (lambda={algo.lambda_})"
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección del Brazo Óptimo', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    
    plt.ylim([0, 105])  # El porcentaje va de 0 a 100; margen superior para legibilidad
    
    plt.tight_layout()
    plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Cota teórica Cte * ln(T).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    
    # Graficar el regret acumulado de cada algoritmo
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    # Si se proporcionan cotas teóricas, graficarlas
    if args:
        for arg_idx, theoretical_bound in enumerate(args):
            if theoretical_bound is not None and len(theoretical_bound) == steps:
                plt.plot(
                    range(steps), 
                    theoretical_bound, 
                    label=f'Cota Teórica {arg_idx + 1}',
                    linestyle='--', 
                    linewidth=2, 
                    alpha=0.7
                )

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Regret Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos y Cotas')
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], optimal_arm_idx: int = None):
    """
    Genera gráficas separadas de estadísticas de brazos para cada algoritmo.
    
    Cada gráfica muestra un histograma donde:
    - Eje X: Cada brazo (con etiqueta indicando número de selecciones y si es óptimo)
    - Eje Y: Recompensa promedio obtenida de cada brazo
    - Barras coloreadas según si el brazo es óptimo o no
    
    :param arm_stats: Lista de diccionarios con estadísticas por algoritmo.
                      Cada diccionario debe contener:
                      {
                          'mean_rewards': np.ndarray,  # Recompensa promedio por brazo
                          'selections': np.ndarray,    # Número de selecciones por brazo
                          'total_reward': np.ndarray,  # Recompensa total por brazo (opcional)
                      }
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param optimal_arm_idx: Índice del brazo óptimo (opcional, se marca en las gráficas).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    
    n_algorithms = len(algorithms)
    n_arms = len(arm_stats[0]['mean_rewards'])
    
    # Crear una figura con subplots (uno por algoritmo)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(7 * n_algorithms, 6))
    
    # Si solo hay un algoritmo, axes no es una lista
    if n_algorithms == 1:
        axes = [axes]
    
    # Colores para barras (óptimo vs subóptimo)
    color_optimal = '#2ecc71'      # Verde brillante
    color_suboptimal = '#3498db'   # Azul
    color_not_selected = '#95a5a6' # Gris para brazos no seleccionados
    
    for algo_idx, (stats, algo, ax) in enumerate(zip(arm_stats, algorithms, axes)):
        mean_rewards = stats['mean_rewards']
        selections = stats['selections']
        
        # Preparar colores para cada barra
        colors = []
        for arm_idx in range(n_arms):
            if selections[arm_idx] == 0:
                colors.append(color_not_selected)
            elif optimal_arm_idx is not None and arm_idx == optimal_arm_idx:
                colors.append(color_optimal)
            else:
                colors.append(color_suboptimal)
        
        # Crear el histograma
        bars = ax.bar(range(n_arms), mean_rewards, color=colors, 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Personalizar etiquetas del eje X - Formato ultra-compacto
        x_labels = []
        for arm_idx in range(n_arms):
            # Formato en una sola línea: B0(1234) o B0★(1234)
            sel_count = int(selections[arm_idx])
            if optimal_arm_idx is not None and arm_idx == optimal_arm_idx:
                label = f"B{arm_idx}★({sel_count})"
            else:
                label = f"B{arm_idx}({sel_count})"
            x_labels.append(label)
        
        ax.set_xticks(range(n_arms))
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        
        # Añadir valores sobre las barras
        for bar, reward, sel in zip(bars, mean_rewards, selections):
            height = bar.get_height()
            if sel > 0:  # Solo mostrar valor si el brazo fue seleccionado
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{reward:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Etiquetas y título
        ax.set_xlabel('Brazos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recompensa Promedio', fontsize=12, fontweight='bold')
        
        algo_label = get_algorithm_label(algo)
        ax.set_title(f'{algo_label}', fontsize=13, fontweight='bold', pad=15)
        
        # Añadir línea horizontal en el valor óptimo si se conoce
        if optimal_arm_idx is not None and selections[optimal_arm_idx] > 0:
            optimal_reward = mean_rewards[optimal_arm_idx]
            ax.axhline(y=optimal_reward, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'Valor óptimo: {optimal_reward:.2f}')
            ax.legend(fontsize=9)
        
        # Ajustar límites del eje Y
        max_reward = np.max(mean_rewards[selections > 0]) if np.any(selections > 0) else 1
        ax.set_ylim([0, max_reward * 1.15])
        
        # Grid
        ax.grid(axis='y', alpha=0.3)
    
    # Título general
    fig.suptitle('Estadísticas de Brazos: Recompensas Promedio y Frecuencia de Selección',
                fontsize=16, fontweight='bold', y=1.02)

    # Leyenda global de colores
    legend_handles = [
        mpatches.Patch(color=color_optimal, label='Brazo óptimo'),
        mpatches.Patch(color=color_suboptimal, label='Brazo subóptimo'),
        mpatches.Patch(color=color_not_selected, label='No seleccionado'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.show()