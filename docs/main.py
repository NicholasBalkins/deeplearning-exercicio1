import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 100 

# dados
class0 = rng.normal([2, 3], [0.8, 2.5], (n, 2))
class1 = rng.normal([5, 6], [1.2, 1.9], (n, 2))
class2 = rng.normal([8, 1], [0.9, 0.9], (n, 2))
class3 = rng.normal([15, 4], [0.5, 2.0], (n, 2))

# scatter
plt.scatter(class0[:,0], class0[:,1], c='red',   s=18, label='Classe 0')
plt.scatter(class1[:,0], class1[:,1], c='blue',  s=18, label='Classe 1')
plt.scatter(class2[:,0], class2[:,1], c='green', s=18, label='Classe 2')
plt.scatter(class3[:,0], class3[:,1], c='purple',s=18, label='Classe 3')

# centróides
means = np.array([
    class0.mean(axis=0),
    class1.mean(axis=0),
    class2.mean(axis=0),
    class3.mean(axis=0),
])

# 3 verticais: pontos médios no eixo X
x01 = (means[0,0] + means[1,0]) / 2
x12 = (means[1,0] + means[2,0]) / 2
x23 = (means[2,0] + means[3,0]) / 2
plt.axvline(x01, color='k', linestyle='--', linewidth=2)
plt.axvline(x12, color='k', linestyle='--', linewidth=2)
plt.axvline(x23, color='k', linestyle='--', linewidth=2)

# 1 horizontal: ponto médio no eixo Y
y02 = (means[0,1] + means[2,1]) / 2
plt.axhline(y02, color='k', linestyle='--', linewidth=2)

plt.legend(loc='best')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Separações 'hard-coded' (4 linhas simples)")
plt.tight_layout()
plt.show()
