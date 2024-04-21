import numpy as np


class Model:
    # 2tN^2 < K
    def __init__(self, g, f, u0, da, b0, length, time, N=10, K=20, force=False):
        self.g = g
        self.f = f
        self.u0 = u0
        self.da = da
        self.b0 = b0
        self.b = length
        self.T = time
        self.N = N
        self.K = K if force else max(K, int(np.ceil(2 * time * (N ** 2))) + 1)
        self.dh = length / N
        self.dt = time / self.K
        self.alpha = self.dt / (self.dh ** 2)
        self.u = np.zeros((self.K, self.N))

    def get_value(self, k, i) -> float:
        k -= 1
        g1 = self.g(self.dh * (i - 0.5))
        g2 = self.g(self.dh * (i + 0.5))
        _x = self.dh * i
        _t = self.dt * k
        return self.u[k][i] + \
               self.alpha * (g1 * self.u[k][i - 1] - (g1 + g2) * self.u[k][i] + g2 * self.u[k][i + 1]) + \
               self.dt * self.f(_x, _t)

    def solve(self):
        u0_vectorized = np.vectorize(self.u0)
        self.u[0] = u0_vectorized(np.linspace(0, self.b, self.N))
        for tf in range(1, self.K):
            self.u[tf][self.N - 1] = self.b0(tf * self.dt)
            for x in range(1, self.N - 1):
                self.u[tf][x] = self.get_value(tf, x)
            self.u[tf][0] = self.da * self.dh + self.u[tf][1]
        return self.u
