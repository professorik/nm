import matplotlib.pyplot as plt
import numpy as np


def print_table(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def heatmap(arr: np.ndarray, length: float = 0, time: float = 0):
    plt.figure('Heatmap')
    if time == 0:
        plt.imshow(arr[::-1], cmap='inferno', vmin=arr.min(), vmax=arr.max())
    else:
        plt.imshow(arr[::-1], cmap='inferno', vmin=arr.min(), vmax=arr.max(), extent=[0, length, 0, time])
    plt.colorbar()


def show_plot(y, length, l='-b+', label=''):
    plt.figure('Temperature')
    plt.plot(np.linspace(0, length, len(y)), y, l, label=label)