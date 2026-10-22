import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load Olivetti Faces Dataset
print("Loading Olivetti Faces dataset...")
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target  # 400 samples, 4096 features

# 2. Train-Test Split (70% train, 30% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Pre-processing: Normalization (fitted STRICTLY on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define dimensions to test (bounded by C-1 = 39 for fair multi-class comparisons)
dimensions = [2, 5, 10, 20, 30, 39]
pca_accuracies = []
lda_accuracies = []
kpca_accuracies = []
valid_dims = []

num_classes = len(np.unique(y_train))
max_dim = num_classes - 1  # 39 for Olivetti Faces

print(
    f"{'Dimension':<10} | {'PCA (%)':<10} | {'LDA (%)':<10} | {'Kernel PCA (%)':<14}"
)
print("-" * 52)

for dim in dimensions:
  if dim > max_dim:
    continue
  valid_dims.append(dim)

  # --- 1. Standard PCA (Unsupervised Linear) ---
  pca = PCA(n_components=dim)
  X_tr_pca = pca.fit_transform(X_train_scaled)
  X_te_pca = pca.transform(X_test_scaled)

  clf_pca = KNeighborsClassifier(n_neighbors=3)
  clf_pca.fit(X_tr_pca, y_train)
  acc_pca = accuracy_score(y_test, clf_pca.predict(X_te_pca)) * 100
  pca_accuracies.append(acc_pca)

  # --- 2. LDA (Supervised Linear) ---
  lda = LDA(n_components=dim)
  X_tr_lda = lda.fit_transform(X_train_scaled, y_train)
  X_te_lda = lda.transform(X_test_scaled)

  clf_lda = KNeighborsClassifier(n_neighbors=3)
  clf_lda.fit(X_tr_lda, y_train)
  acc_lda = accuracy_score(y_test, clf_lda.predict(X_te_lda)) * 100
  lda_accuracies.append(acc_lda)

  # --- 3. Kernel PCA (Unsupervised Non-Linear with RBF Kernel) ---
  kpca = KernelPCA(n_components=dim, kernel="rbf", gamma=0.01, fit_inverse_transform=False, random_state=42)
  X_tr_kpca = kpca.fit_transform(X_train_scaled)
  X_te_kpca = kpca.transform(X_test_scaled)

  clf_kpca = KNeighborsClassifier(n_neighbors=3)
  clf_kpca.fit(X_tr_kpca, y_train)
  acc_kpca = accuracy_score(y_test, clf_kpca.predict(X_te_kpca)) * 100
  kpca_accuracies.append(acc_kpca)

  print(
      f"{dim:<10} | {acc_pca:<10.2f} | {acc_lda:<10.2f} | {acc_kpca:<14.2f}"
  )

# 4. Plotting Comparison Curve
plt.figure(figsize=(9, 5))
plt.plot(
    valid_dims,
    pca_accuracies,
    marker="o",
    linestyle="-",
    color="blue",
    label="Standard PCA + k-NN",
    linewidth=2,
)
plt.plot(
    valid_dims,
    lda_accuracies,
    marker="s",
    linestyle="--",
    color="orange",
    label="LDA + k-NN",
    linewidth=2,
)
plt.plot(
    valid_dims,
    kpca_accuracies,
    marker="^",
    linestyle="-.",
    color="purple",
    label="Kernel PCA (RBF) + k-NN",
    linewidth=2,
)

plt.title(
    "Dimensionality Reduction Comparison: Linear PCA vs. LDA vs. Kernel PCA"
)
plt.xlabel("Number of Dimensions")
plt.ylabel("Classification Accuracy (%)")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("pca_vs_lda_vs_kpca.png", dpi=300)
plt.show()