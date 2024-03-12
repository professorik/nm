import utils
import projection_model as pm
import projection_model_2 as pm_2
import numpy as np
import time


def g(x):
    return np.exp(x)


def f(x):
    return x**2


def ans(x):
    return 3 - np.exp(-x) * (x**3 + 3*(x**2) + 6*x + 6) / 3


if __name__ == '__main__':
    N = 20
    model = pm.Model(g, f, v1=(0, 1), v2=(1, 1), N=N)
    model.solve()
    model_2 = pm_2.Model(g, f, v1=(0, 1), v2=(1, 1), N=N)
    model_2.solve()

    #utils.show_plot(ans(np.linspace(0, 1, N)), length=1, l='-g+', label='Original')
    utils.show_plot(model.get_value(np.linspace(0, 1, N)), length=1, l='-r+', label='$\\varphi_k=sin(xk\pi)$')
    utils.show_plot(model_2.get_value(np.linspace(0, 1, N)), length=1, label='linear bf')
    utils.plt.legend()
    utils.plt.show()
