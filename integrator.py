import numpy as np
import scipy


def int_rect(f, a, b, delta):
    x = np.arange(a, b, delta)
    f_vectorized = np.vectorize(f)
    return f_vectorized(x).sum() * delta


if __name__ == '__main__':
    print(int_rect(lambda x: x**2, 1, 4, 1e-6))
    print(scipy.integrate.quad(lambda x: x**2, 1, 4))