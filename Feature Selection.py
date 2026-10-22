import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load Dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target  # 400 images, 4096 pixels each

# 2. Train-Test Split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Standardization (fitted ONLY on training set)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define feature counts to evaluate
k_features_list = [5, 10, 20, 39, 80, 150, 300, 500]
# Calculate each pixel's mutual information once, then reuse the ranking.
print("Calculating mutual information for 4096 pixels...", flush=True)
mi_scores = mutual_info_classif(
    X_train_scaled, y_train, random_state=42, n_jobs=-1
)
ranked_features = np.argsort(mi_scores)[::-1]
mi_accuracies = []

print(f"{'Selected Features (k)':<22} | {'Mutual Info + k-NN Accuracy':<30}")
print("-" * 55)

for k in k_features_list:
  print(f"Evaluating k={k}...", flush=True)
  selected_features = ranked_features[:k]
  X_train_selected = X_train_scaled[:, selected_features]
  X_test_selected = X_test_scaled[:, selected_features]

  clf = KNeighborsClassifier(n_neighbors=3)
  clf.fit(X_train_selected, y_train)
  acc = accuracy_score(y_test, clf.predict(X_test_selected)) * 100
  mi_accuracies.append(acc)

  print(f"{k:<22} | {acc:.2f}%")

# Plotting Feature Selection Curve
plt.figure(figsize=(8, 5))
plt.plot(
    k_features_list,
    mi_accuracies,
    marker="^",
    linestyle="-",
    color="green",
    linewidth=2,
    label="Filter (Mutual Info) + k-NN",
)
plt.title("Feature Selection: Classification Accuracy vs. Number of Pixels Selected")
plt.xlabel("Number of Selected Features (Pixels)")
plt.ylabel("Classification Accuracy (%)")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("feature_selection_results.png", dpi=300)
plt.show()