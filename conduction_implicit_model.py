import numpy as np


def TDMA(lower, main, upper, res):
    n = len(res)
    a = np.zeros(n - 1, float)
    b = np.zeros(n, float)
    x = np.zeros(n, float)

    a[0] = upper[0] / main[0]
    b[0] = res[0] / main[0]

    for i in range(1, n - 1):
        a[i] = upper[i] / (main[i] - lower[i - 1] * a[i - 1])
    for i in range(1, n):
        b[i] = (res[i] - lower[i - 1] * b[i - 1]) / (main[i] - lower[i - 1] * a[i - 1])
    x[n - 1] = b[n - 1]
    for i in range(n - 1, 0, -1):
        x[i - 1] = b[i - 1] - a[i - 1] * x[i]
    return x


class Model:
    def __init__(self, u0, a0, b0, length, time, N=10, K=20):
        self.u0 = u0
        self.a0 = a0
        self.b0 = b0
        self.b = length
        self.N = N
        self.K = K
        self.dh = length / N
        self.dhs = self.dh ** 2
        self.dt = time / self.K
        self.u = np.zeros((self.K, self.N))

    def get_res(self, k):
        res: list = [self.u[k - 1][1] / self.dt + self.u[k][0] / self.dhs]
        res.extend(self.u[k - 1][2:-2] / self.dt)
        res.append(self.u[k - 1][-2] / self.dt + self.u[k][-1] / self.dhs)
        return res

    def solve(self):
        upper = [-1 / self.dhs] * (self.N - 3)
        main = [1 / self.dt + 2 / self.dhs] * (self.N - 2)
        lower = upper.copy()

        u0_vectorized = np.vectorize(self.u0)
        self.u[0] = u0_vectorized(np.linspace(0, self.b, self.N))

        for tf in range(1, self.K):
            self.u[tf][0] = self.a0(tf * self.dt)
            self.u[tf][self.N - 1] = self.b0(tf * self.dt)
            solution = TDMA(lower, main, upper, self.get_res(tf))
            for x in range(1, self.N - 1):
                self.u[tf][x] = solution[x - 1]
        return self.u
