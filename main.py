import utils
import conduction_model as cm
import conduction_implicit_model as cim
import numpy as np
import time


def g(x) -> float:
    return 1


def f(x, t) -> float:
    return 0


def u0(x):
    return np.sin(x * np.pi)


def a0(t):
    return np.sin(t * np.pi / 2)


def b0(t):
    return a0(t)


def problem_1():
    start = time.time()
    model = cm.Model(g, f, u0, a0, b0, length=1, time=1, N=50, K=-1)
    table = model.solve()
    #model_2 = cm.Model(g, f, u0, a0, b0, length=1, time=1, N=100, K=-1)
    #table_2 = model_2.solve()
    # Prints 8.676048278808594
    print(time.time() - start)

    #utils.heatmap(table, length=1, time=1)
    #utils.show_plot(table[0], length=1, l='-g+', label='Initial values')
    utils.show_plot(table[-1], length=1, label='50 points')
    #utils.show_plot(table_2[-1], length=1, l='-r+', label='100 points')
    # utils.print_table(table)


def problem_2():
    start = time.time()
    model = cim.Model(u0, a0, b0, length=1, time=1, N=50, K=100)
    table = model.solve()
    #model_2 = cim.Model(u0, a0, b0, length=1, time=1, N=100, K=100)
    #table_2 = model_2.solve()
    # Prints 0.03999686241149902
    print(time.time() - start)

    #utils.heatmap(table, length=1, time=1)
    #utils.show_plot(table[0], length=1, l='-g+', label='Initial values')
    utils.show_plot(table[-1], length=1, label='100 points', l='-r+')
    #utils.show_plot(table_2[-1], length=1, l='-r+', label='100 points')
    # utils.print_table(table)


if __name__ == '__main__':
    problem_1()
    problem_2()
    utils.plt.legend()
    utils.plt.show()
