import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
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

# Define dimensions to test (bounded by C-1 = 39 for multi-class LDA/NCA comparisons)
dimensions = [2, 5, 10, 20, 30, 39]
pca_accuracies = []
lda_accuracies = []
nca_accuracies = []
valid_dims = []

num_classes = len(np.unique(y_train))
max_dim = num_classes - 1  # 39 for Olivetti Faces

print(
    f"{'Dimension':<10} | {'PCA (%)':<10} | {'LDA (%)':<10} | {'NCA (%)':<10}"
)
print("-" * 48)

for dim in dimensions:
  if dim > max_dim:
    continue
  valid_dims.append(dim)

  # --- 1. PCA (Unsupervised Linear) ---
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

  # --- 3. NCA (Supervised Metric/Subspace Learning for k-NN) ---
  nca = NeighborhoodComponentsAnalysis(n_components=dim, random_state=42)
  X_tr_nca = nca.fit_transform(X_train_scaled, y_train)
  X_te_nca = nca.transform(X_test_scaled)

  clf_nca = KNeighborsClassifier(n_neighbors=3)
  clf_nca.fit(X_tr_nca, y_train)
  acc_nca = accuracy_score(y_test, clf_nca.predict(X_te_nca)) * 100
  nca_accuracies.append(acc_nca)

  print(
      f"{dim:<10} | {acc_pca:<10.2f} | {acc_lda:<10.2f} | {acc_nca:<10.2f}"
  )

# 4. Plotting Comparison Curve
plt.figure(figsize=(9, 5))
plt.plot(
    valid_dims,
    pca_accuracies,
    marker="o",
    linestyle="-",
    color="blue",
    label="PCA + k-NN",
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
    nca_accuracies,
    marker="^",
    linestyle="-.",
    color="green",
    label="NCA + k-NN",
    linewidth=2,
)

plt.title(
    "Dimensionality Reduction Comparison: PCA vs. LDA vs. NCA (k-NN Classifier)"
)
plt.xlabel("Number of Dimensions")
plt.ylabel("Classification Accuracy (%)")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("pca_vs_lda_vs_nca.png", dpi=300)
plt.show()