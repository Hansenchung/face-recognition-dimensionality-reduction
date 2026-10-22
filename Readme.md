# Comparative Analysis of Dimensionality Reduction and Feature Selection for Face Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository explores and benchmarks various **Dimensionality Reduction (DR)** and **Feature Selection** techniques for multi-class face recognition on the **Olivetti Faces dataset**[cite: 6, 7]. Using a $k$-Nearest Neighbors ($k$-NN, $k=3$) classifier[cite: 4, 5, 6, 7], the project evaluates how different linear, non-linear, and supervised/unsupervised algorithms scale in classification accuracy as the number of retained dimensions/features changes.

---

## 📊 Key Experimental Comparisons

All experiments use a stratified **70% train / 30% test split** (400 total samples, 40 classes, 4096 pixels per image) with features strictly standardized on the training split[cite: 4, 5, 6, 7].

### 1. Linear Unsupervised vs. Supervised: PCA (Eigenfaces) vs. LDA (Fisherfaces)
* **Standard PCA (Eigenfaces)** is an unsupervised linear technique maximizing global variance.
* **Linear Discriminant Analysis (LDA / Fisherfaces)** is a supervised technique maximizing between-class variance while minimizing within-class variance (bounded by $C - 1 = 39$ dimensions for 40 classes)[cite: 6].
* **Result:** LDA significantly outperforms PCA at lower dimensions, achieving near-optimal accuracy (~97-98%) with far fewer components[cite: 1, 6].

<p align="center">
  <img src="outputs/faces_pca_vs_lda.png" width="700" alt="PCA vs LDA Performance">
</p>

### 2. Introducing Non-Linearity: Kernel PCA (RBF)
* **Kernel PCA** applies the kernel trick to map data into a higher-dimensional feature space for non-linear dimensionality reduction.
* **Result:** Standard Kernel PCA with an RBF kernel struggles significantly under the multi-class nearest neighbor pipeline on pixel intensity data, yielding flat near-zero accuracy compared to linear baselines[cite: 2, 4].

<p align="center">
  <img src="outputs/pca_vs_lda_vs_kpca.png" width="700" alt="PCA vs LDA vs Kernel PCA">
</p>

### 3. Metric Learning / Supervised Subspace: Neighborhood Components Analysis (NCA)
* **NCA** is a non-parametric supervised method that learns a linear transformation to maximize k-NN classification performance.
* **Result:** NCA matches or slightly exceeds LDA performance at higher dimensions (approaching ~98-99% accuracy)[cite: 3, 5].

<p align="center">
  <img src="outputs/pca_vs_lda_vs_nca.png" width="700" alt="PCA vs LDA vs NCA">
</p>

### 4. Filter-Based Feature Selection: Mutual Information
* Instead of projecting data into a lower-dimensional subspace, **Mutual Information** ranks and selects the most informative individual pixels directly[cite: 7].
* **Result:** Proves that raw pixel selection can also capture discriminatory regions effectively when scaled to hundreds of features[cite: 7].

<p align="center">
  <img src="outputs/feature_selection_results.png" width="700" alt="Feature Selection Results">
</p>

---

## 🛠️ Installation & Requirements

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/face-recognition-dimensionality-reduction.git](https://github.com/your-username/face-recognition-dimensionality-reduction.git)
cd face-recognition-dimensionality-reduction
pip install -r requirements.txt