import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load Olivetti Faces Dataset
print("Loading Olivetti Faces dataset...")
data = fetch_olivetti_faces(
    data_home=r"C:\Users\hanse\scikit-learn-data",
    download_if_missing=True,
    shuffle=True,
    random_state=42,
)
X, y = data.data, data.target  # X shape: (400, 4096), y shape: (400,)

# 2. Train-Test Split (70% train, 30% test, stratified by person)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Pre-processing: Normalization (fitted strictly on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define dimensions to test (ranging from low to higher dimensional spaces)
dimensions = [2, 5, 10, 20, 30, 39, 50, 80, 100, 150]
pca_accuracies = []
lda_accuracies = []
valid_dims = []

num_classes = len(np.unique(y_train))  # 40 classes
max_lda_dim = num_classes - 1  # 39 max dimensions for LDA

print(
    f"{'Dimension':<10} | {'PCA Accuracy (%)':<16} | {'LDA Accuracy (%)':<16}"
)
print("-" * 50)

for dim in dimensions:
  if dim > X_train.shape[1]:
    continue
  valid_dims.append(dim)

  # --- PCA Pipeline (Eigenfaces) ---
  pca = PCA(n_components=dim)
  X_train_pca = pca.fit_transform(X_train_scaled)
  X_test_pca = pca.transform(X_test_scaled)

  clf_pca = KNeighborsClassifier(n_neighbors=3)
  clf_pca.fit(X_train_pca, y_train)
  acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca)) * 100
  pca_accuracies.append(acc_pca)

  # --- LDA Pipeline (Fisherfaces - constrained by C-1) ---
  if dim <= max_lda_dim:
    lda = LDA(n_components=dim)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    clf_lda = KNeighborsClassifier(n_neighbors=3)
    clf_lda.fit(X_train_lda, y_train)
    acc_lda = accuracy_score(y_test, clf_lda.predict(X_test_lda)) * 100
    lda_accuracies.append(acc_lda)
    lda_str = f"{acc_lda:.2f}%"
  else:
    lda_accuracies.append(None)
    lda_str = "N/A (Max C-1)"

  print(f"{dim:<10} | {acc_pca:.2f}%          | {lda_str}")

# 4. Plotting Comparison Curve
plt.figure(figsize=(9, 5))
lda_x = [valid_dims[i] for i, val in enumerate(lda_accuracies) if val is not None]
lda_y = [val for val in lda_accuracies if val is not None]

plt.plot(
    valid_dims,
    pca_accuracies,
    marker="o",
    linestyle="-",
    color="blue",
    label="PCA (Eigenfaces) + k-NN",
    linewidth=2,
)
plt.plot(
    lda_x,
    lda_y,
    marker="s",
    linestyle="--",
    color="orange",
    label="LDA (Fisherfaces) + k-NN",
    linewidth=2,
)

plt.title("Face Recognition: PCA vs. LDA Performance vs. Dimensionality")
plt.xlabel("Number of Dimensions")
plt.ylabel("Classification Accuracy (%)")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("faces_pca_vs_lda.png", dpi=300)
plt.show()