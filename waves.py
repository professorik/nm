import scipy

import utils
import numpy as np
import time


def t(n, j):
    return np.cos(j * np.pi / n)


def T(r, k, n):
    #return np.cos(r * np.arccos(t(n, k)))
    return np.cos(r * k * np.pi / n)


class Model:
    def __init__(self, beta, d, L, _lambda, K=200, N=50):
        self.angle = beta
        self.k0 = 2 * np.pi / _lambda
        self.d = d
        self.L = L
        self._lambda = _lambda
        self.N = N
        self.K = K
        self.kappa = self.L * self.k0 / (2 * np.pi)
        self.gv = {}
        self.bv = {}
        self.alpha = -d/2
        self.beta = d/2
        self.B1 = 1j
        self.B2 = (1j * self.kappa**2) / 2
        self.gamma = np.zeros(self.N - 1, dtype='complex_')

    def get_m_i(self, i):
        tmp = 0
        for k in range(1, self.N):
            tt = t(self.N, k)
            tmp += self.gamma[k - 1] * (1 - tt**2) * np.exp(-1j * i * tt * self.d / 2)
        return self.d * tmp / (4 * self.N)

    def get_M(self, n):
        res = np.zeros(n, dtype='complex_')
        lower = -(n - 1)//2
        for i in range(n):
            res[i] = self.get_m_i(i + lower)
        return res

    def fbeta(self, n):
        return 2 * np.pi * n / self.L

    def fgamma(self, n):
        tmp = self.bv.get(n, self.fbeta(n))
        return np.sqrt(self.k0 ** 2 - tmp ** 2 + 0j)

    def K1(self, theta, phi):
        res = 0
        delta = phi - theta
        for i in range(1, self.N + 1):
            res += (self.gv[i] - self.B1*i - self.B2/i) * np.cos(i * delta)
        return res

    def g(self, _t):
        return ((self.beta - self.alpha) * _t + self.beta + self.alpha) / 2

    def K0(self, t0, t1):
        gt0 = self.g(t0)
        gt = self.g(t1)
        tmp = (gt0 - gt) / 2
        A = 0
        for k in range(-50, 51):
            if k == 0: continue
            A += (tmp - k * np.pi)**(-2)
        A *= self.B1/4
        B = 0 if tmp == 0 else self.B2 * np.log(np.abs(np.sin(tmp)/tmp))
        return self.K1(gt0, gt) - A - B

    def get_matrix(self):
        matrix = np.zeros((self.N - 1, self.N - 1), dtype='complex_')
        for j in range(1, self.N):
            tmp = self.B1 / self.N
            cache = {}
            for k in range(1, self.N):
                cache[k] = (1 - t(self.N, k)**2)
                if (j + k) % 2 == 0: continue
                matrix[j-1][k-1] += 2 * tmp * cache[k] / (t(self.N, j)-t(self.N, k))**2
            matrix[j-1][j-1] -= self.B1 * self.N / 2
            tmp = (self.B2 / 4 * self.N) * (self.d/2)**2
            for k in range(1, self.N):
                tmp_sum = 0
                for r in range(1, self.N):
                    tmp_sum += T(r, k, self.N) * T(r, j, self.N) / r
                matrix[j-1][k-1] -= tmp * cache[k] * (np.log(2)+2*tmp_sum+((-1)**(k+j)) / (2 * self.N))
            tmp = (self.d**2) / (2 * self.N)
            for k in range(1, self.N):
                matrix[j-1][k-1] -= tmp * cache[k] * self.K0(t(self.N, j), t(self.N, k))
        return matrix

    # FIXME: sus
    def f(self, t0):
        return self.kappa

    def solve_lae(self):
        G = self.get_matrix()
        d = np.zeros(self.N - 1)
        for j in range(1, self.N):
            d[j-1] = -self.f(t(self.N - 2, j))
        return scipy.linalg.solve(G, d)

    def get_u_v(self, x, y):
        res = [0, 0]
        for i in range(-self.N, self.N + 1):
            if y > 0:
                res[0] += (self.M[i + self.N] - int(not i)) * \
                          np.exp(1j * (self.bv[i] * x + self.gv[i] * y))
            res[1] += self.M[i + self.N] * np.exp(1j * (self.bv[i] * x - self.gv[i] * y))
        return np.abs(res[0]), np.abs(res[1])

    def solve(self, w=1, h=1):
        self.u = np.zeros((2 * self.K + 1, 2 * self.N + 1))
        for i in range(-self.N, self.N + 1):
            self.bv[i] = self.fbeta(i)
            self.gv[i] = self.fgamma(i)
        self.gamma = self.solve_lae()
        self.M = self.get_M(2 * self.N + 1)
        X = np.linspace(-w * self.L / 2, w * self.L / 2, 2 * self.N + 1)
        Y = np.linspace(-h * 2 * self.L, h * 2 * self.L, 2 * self.K + 1)
        for yi in range(len(Y) // 2, len(Y)):
            for xi in range(len(X)):
                self.u[yi][xi], self.u[-yi][xi] = self.get_u_v(X[xi], Y[yi])
        return self.u


if __name__ == '__main__':
    start = time.time()
    model = Model(beta=np.pi / 2, d=0.01, L=0.05, _lambda=0.0005, K=120, N=50)
    table = model.solve(w=4)
    print(time.time() - start)

    utils.heatmap_waves(table, w=4*0.05, h=0.2)
    #utils.plt.legend()
    utils.plt.show()
