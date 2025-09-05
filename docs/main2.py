import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------- 1) Dados 5D ----------
n = 500

mu_A = np.array([0, 0, 0, 0, 0], dtype=float)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=float)
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

A = rng.multivariate_normal(mu_A, Sigma_A, size=n)
B = rng.multivariate_normal(mu_B, Sigma_B, size=n)

X = np.vstack([A, B])
y = np.array([0]*n + [1]*n) 

Xc = X - X.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
V = Vt.T
Z = Xc @ V[:, :2]   # (1000, 2)

# ---------- 2) Linha entre os grupos ----------
# pega a média de PC1 para cada classe
mA = Z[y==0,0].mean()
mB = Z[y==1,0].mean()
x_sep = (mA + mB) / 2  # ponto médio no eixo X

# ---------- 3) Plot ----------
plt.figure(figsize=(6,5))
plt.scatter(Z[y==0, 0], Z[y==0, 1], s=12, label="Classe A")
plt.scatter(Z[y==1, 0], Z[y==1, 1], s=12, label="Classe B")

# linha vertical separando
plt.axvline(x=x_sep, color='k', linestyle='--', linewidth=2, label="Linha de separação")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (5D → 2D) de A vs B com linha entre os grupos")
plt.legend()
plt.tight_layout()
plt.show()
