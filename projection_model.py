import numpy as np
import scipy


class Model:
    def __init__(self, g, f, v1, v2, N):
        self.g = g
        self.f = f
        self.N = N
        self.a0 = v1[1]
        self.b0 = v2[1]
        self.beta0 = (v2[0] * self.a0 - v1[0] * self.b0) / (v2[0] - v1[0])
        if v1[0] != 0:
            self.beta1 = (self.a0 + (v1[0] - 1) * self.beta0) / v1[0]
        else:
            self.beta1 = (self.b0 + (v2[0] - 1) * self.beta0) / v2[0]
        self.phi_0 = lambda x: self.beta0 + (self.beta1 - self.beta0) * x
        self.d_phi_0 = lambda x: self.beta1 - self.beta0
        self.phi_k = lambda x, k: np.sin(k * x * np.pi)
        self.d_phi_k = lambda x, k: k * np.pi * np.cos(k * x * np.pi)

    def solve(self, delta=0):
        d = np.zeros(self.N)
        C = np.zeros((self.N, self.N))
        for i in range(1, self.N + 1):
            d[i - 1] = scipy.integrate.quad(
                lambda x: self.f(x) * self.phi_k(x, i) - self.g(x) * self.d_phi_0(x) * self.d_phi_k(x, i),
                0, 1
            )[0]
            for k in range(1, self.N + 1):
                C[i - 1][k - 1] = scipy.integrate.quad(
                    lambda x: self.g(x) * self.d_phi_k(x, k) * self.d_phi_k(x, i),
                    0, 1
                )[0]
        self.a = scipy.linalg.solve(-C + delta, d)

    def get_value(self, x):
        res = self.phi_0(x)
        for k in range(1, self.N + 1):
            res += self.a[k - 1] * self.phi_k(x, k)
        return res
