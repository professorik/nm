import utils
import waves
import numpy as np
import time


def problem_1():
    start = time.time()
    model = waves.Model(beta=0, d=0.01, L=0.05, _lambda=0.0005)
    table = model.solve()
    # Prints 9.776944875717163
    print(time.time() - start)

    utils.heatmap(table, length=1, time=1)
    #utils.show_plot(table[0], length=1, l='-g+', label='Initial values')
    #utils.show_plot(table[-1], length=1, label='50 points')
    #utils.show_plot(table_2[-1], length=1, l='-r+', label='100 points')
    utils.plt.legend()
    utils.plt.show()
    # utils.print_table(table)


if __name__ == '__main__':
    problem_1()