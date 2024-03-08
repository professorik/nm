import numpy as np
import scipy
import projection_model as pm


class Model(pm.Model):
    def __init__(self, g, f, v1, v2, N):
        super().__init__(g, f, v1, v2, N)
        self.xk = np.linspace(v1[0], v2[0], self.N)
        self.h = (v2[0] - v1[0]) / self.N
        self.ih = 1 / self.h
        self.d_phi_k = lambda x, k: self._d_phi_k(x, k)
        self.phi_k = lambda x, k: self._phi_k(x, k)

    def _phi_k(self, x, k):
        k -= 1
        if np.abs(x - self.xk[k]) > self.h:
            return 0
        if self.xk[k - 1] <= x <= self.xk[k]:
            return (x - self.xk[k - 1]) * self.ih
        return (self.xk[k + 1] - x) * self.ih

    def _d_phi_k(self, x, k):
        k -= 1
        if np.abs(x - self.xk[k]) > self.h:
            return 0
        if self.xk[k] <= x <= self.xk[k + 1]:
            return -self.ih
        return self.ih

    def solve(self):
        super().solve((self.g(1) - self.g(0)) * self.ih)

    def get_value(self, x):
        res = self.phi_0(x)
        for xi in x:
            for k in range(1, self.N + 1):
                res += self.a[k - 1] * self.phi_k(xi, k)
        return res