import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 1. Carregar dados ----------
train = pd.read_csv("deeplearning-exercicio1/docs/train.csv")

# ---------- 2. Tratar clunas ----------
# Separar alvo
y = train["Transported"].astype(int)  # True/False -> 1/0
X = train.drop(columns=["Transported", "PassengerId", "Name"])

# Dividir Cabin em 3 partes (Deck, Num, Side)
cabin_split = X["Cabin"].str.split("/", expand=True)
X["Deck"] = cabin_split[0]
X["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
X["Side"] = cabin_split[2]
X = X.drop(columns=["Cabin"])

# ---------- 3. Tratar data faltando ----------
for col in X.select_dtypes(include=["float64", "int64"]).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=["object", "bool"]).columns:
    X[col] = X[col].fillna("Missing")

# ---------- 4. One-hot encoding ----------
X = pd.get_dummies(X, drop_first=True)

# ---------- 5. Normalização ----------
scaler = StandardScaler()
num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]
X[num_cols] = scaler.fit_transform(X[num_cols])

# ---------- 6. Visualização ----------
fig, axes = plt.subplots(1,2, figsize=(10,4))
train["Age"].hist(ax=axes[0], bins=30)
axes[0].set_title("Age original")
X["Age"].hist(ax=axes[1], bins=30)
axes[1].set_title("Age padronizado")
plt.show()

print("Shape final:", X.shape)
