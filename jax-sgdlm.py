#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 23:41:22 2025

@author: lyh2019
"""

# sgdlm_jax.py
# Python replication of Gruber and West (2016) with JAX support

import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import block_diag
from jax.scipy.special import digamma
from jax import jit, vmap
from jax.scipy.stats import multivariate_normal
import numpy as np
import pandas as pd


def draw_gamma(key, shape, scale, size):
    return random.gamma(key, shape, shape=(size,)) * scale


def mvn_sample(key, mean, cov, N):
    L = jnp.linalg.cholesky(cov)
    z = random.normal(key, (N, mean.size))
    return mean + z @ L.T


class UnivariateDLM:
    def __init__(self, p, delta_phi, delta_gamma, beta):
        self.p = p
        self.delta_phi = delta_phi
        self.delta_gamma = delta_gamma
        self.beta = beta

        self.a = jnp.zeros((p + 1, 1))
        self.R = jnp.diag(jnp.array([0.0001] + [0.01] * p))
        self.r = 5.0
        self.c = 0.001

    def kalman_update(self, F, y):
        f = F.T @ self.a
        q = F.T @ self.R @ F + self.c
        e = y - f
        A = self.R @ F / q
        z = (self.r + e**2 / q) / (self.r + 1)

        m = self.a + A * e
        C = (self.R - A @ A.T * q) * z
        n = self.r + 1
        s = z * self.c

        self.m = m
        self.C = C
        self.s = s
        self.n = n

        return m, C, s, n

    def evolve(self):
        block_1 = self.C[0:1, 0:1] * (1 / self.delta_phi - 1)
        block_2 = self.C[1:, 1:] * (1 / self.delta_gamma - 1)
        W = block_diag((block_1, block_2))
        self.R = self.C + W
        self.a = self.m
        self.c = self.s
        self.r = self.beta * self.n


import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate data
T = 100
theta_true = jnp.array([1.0, 2.0])  # intercept + slope
x = jnp.linspace(0, 1, T)
X = jnp.stack([jnp.ones(T), x], axis=1)  # add intercept
y = X @ theta_true + 0.1 * jnp.array(np.random.randn(T))  # add noise

# 2. Initialize DLM
p_own = 1  # 1 predictor (excluding intercept)
p_parent = 0  # no parents
delta_phi = 0.99
delta_gamma = 0.95

m_init = jnp.zeros((2, 1))            # intercept + 1 slope
C_init = jnp.eye(2) * 0.01
s_init = 1.0
n_init = 5.0

dlm = UnivariateDLM(m_init, C_init, s_init, n_init, delta_phi, delta_gamma, p_own, p_parent)

# 3. Run recursive filter
m_trace = []

for t in range(T):
    F_t = X[t].reshape(-1, 1)  # shape (2,1)
    dlm.kalman_update(F_t, y[t])
    m_trace.append(dlm.m.flatten())
    dlm.evolve()

# 4. Plot results
m_trace = jnp.array(m_trace)
plt.plot(m_trace[:, 0], label="Estimated Intercept")
plt.plot(m_trace[:, 1], label="Estimated Slope")
plt.axhline(theta_true[0], linestyle="--", color="gray", label="True Intercept")
plt.axhline(theta_true[1], linestyle="--", color="black", label="True Slope")
plt.title("Posterior Means from UnivariateDLM")
plt.legend()
plt.show()






class SGDLM:
    def __init__(self, m, p, delta_phi, delta_gamma, beta, N, seed=0):
        self.m = m
        self.p = p
        self.N = N
        self.models = [UnivariateDLM(p, delta_phi, delta_gamma, beta) for _ in range(m)]
        self.key = random.PRNGKey(seed)

    def importance_sampling(self, m, C, s, n):
        key1, key2 = random.split(self.key)
        lambdas = draw_gamma(key1, n / 2, 2 / (n * s), self.N)
        covs = jnp.array([C / (l * s) for l in lambdas])
        thetas = mvn_sample(key2, m.flatten(), C / (lambdas[0] * s), self.N)
        return thetas.reshape(self.N, self.p + 1, 1), lambdas

    def compute_weights(self, thetas, parent_map):
        def get_gamma(theta):
            Gamma = jnp.zeros((self.m, self.m))
            for j in range(self.m):
                for idx, k in enumerate(parent_map[j]):
                    Gamma = Gamma.at[j, k].set(theta[j, idx + 1, 0])
            return jnp.linalg.det(jnp.eye(self.m) - Gamma)

        return jnp.abs(vmap(get_gamma)(thetas))

    def vb_update(self, thetas, lambdas, weights):
        alpha = weights / jnp.sum(weights)
        m = jnp.sum(alpha[:, None, None] * lambdas[:, None, None] * thetas, axis=0) / jnp.sum(alpha * lambdas)
        V = jnp.sum(alpha[:, None, None] * lambdas[:, None, None] *
                    (thetas - m) @ jnp.transpose((thetas - m), (0, 2, 1)), axis=0)

        Vinv = jnp.linalg.pinv(V)
        d = jnp.sum(alpha * lambdas * vmap(lambda t: (t - m).T @ Vinv @ (t - m))(thetas).squeeze())

        def root_func(n):
            return (jnp.log(n + self.p + 1 - d) - digamma(n / 2) - (self.p + 1 - d) / n -
                    jnp.log(2 * jnp.sum(alpha * lambdas)) + jnp.sum(alpha * jnp.log(lambdas)))

        n_grid = jnp.linspace(2.1, 1000, 1000)
        f_vals = vmap(root_func)(n_grid)
        n_opt = n_grid[jnp.argmin(jnp.abs(f_vals))]

        s = (n_opt + self.p + 1 - d) / (n_opt * jnp.sum(alpha * lambdas))
        C = s * V
        return m, C, s, n_opt

    def step(self, Y_t, F_t_list, parent_map):
        posteriors = [self.models[j].kalman_update(F_t_list[j], Y_t[j]) for j in range(self.m)]
        thetas_list, lambdas_list = [], []

        for j in range(self.m):
            theta_j, lambda_j = self.importance_sampling(*posteriors[j])
            thetas_list.append(theta_j)
            lambdas_list.append(lambda_j)

        stacked_thetas = jnp.stack([t.squeeze() for t in thetas_list], axis=1)  # shape: (N, m, p+1)
        weights = self.compute_weights(stacked_thetas, parent_map)

        for j in range(self.m):
            m, C, s, n = self.vb_update(thetas_list[j], lambdas_list[j], weights)
            self.models[j].m, self.models[j].C = m, C
            self.models[j].s, self.models[j].n = s, n
            self.models[j].evolve()


if __name__ == "__main__":
    T, m, p = 200, 5, 2
    np.random.seed(42)

    true_states = np.zeros((T, m))
    observations = np.zeros((T, m))
    for t in range(1, T):
        true_states[t] = 0.8 * true_states[t-1] + np.random.normal(0, 0.5, m)
    for t in range(T):
        for i in range(m):
            parents = [j for j in range(m) if j != i][:p]
            obs = 0.5 + true_states[t, i] + sum(0.3 * true_states[t, j] for j in parents)
            observations[t, i] = obs + np.random.normal(0, 0.2)

    Y = pd.DataFrame(observations, columns=[f"y{i+1}" for i in range(m)])
    F_t_list = [[jnp.array([[1.0] + [Y.iloc[t, j] for j in range(m) if j != i][:p]]).T for i in range(m)] for t in range(T)]
    parent_map = {i: [j for j in range(m) if j != i][:p] for i in range(m)}

    model = SGDLM(m=m, p=p, delta_phi=0.99, delta_gamma=0.95, beta=0.9, N=100)
    for t in range(T):
        y_t = jnp.array(Y.iloc[t].values).reshape(-1, 1)
        model.step(y_t, F_t_list[t], parent_map)

    final_means = jnp.array([mod.m.flatten() for mod in model.models])
    print("\nFinal Posterior Means (JAX version):\n")
    print(pd.DataFrame(np.array(final_means), columns=[f"theta_{i}" for i in range(p+1)]))
