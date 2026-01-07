# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# =========================
# 2. LOAD DATASET
# =========================
# OPTION 1: CSV
# df = pd.read_csv("dataset.csv")

# OPTION 2: Built-in dataset (example: Iris)
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print(df.head())
print(df.info())


# =========================
# 3. HANDLE CATEGORICAL FEATURES (IF ANY)
# =========================
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])


# =========================
# 4. FEATURE MATRIX (NO TARGET IN UNSUPERVISED)
# =========================
X_raw = df.select_dtypes(include=np.number)


# =========================
# 5. FEATURE SCALING (MANDATORY)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


# =========================
# 6. ELBOW METHOD (K SELECTION)
# =========================
wcss = []

for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 10), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()


# =========================
# 7. TRAIN FINAL MODEL
# =========================
K = 3   # 🔁 SET AFTER ELBOW

km = KMeans(
    n_clusters=K,
    init="k-means++",
    max_iter=300,
    n_init=10,
    random_state=42
)

km.fit(X_scaled)


# =========================
# 8. OUTPUTS
# =========================
labels = km.labels_
centroids = km.cluster_centers_

print("Cluster Labels:\n", labels)
print("Centroids:\n", centroids)


# =========================
# 9. EVALUATION
# =========================
sil_score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", sil_score)


# =========================
# 10. VISUALIZATION
# =========================
# If features > 2 → use PCA automatically
if X_scaled.shape[1] > 2:
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_scaled)
    x_label, y_label = "PCA Component 1", "PCA Component 2"
else:
    X_vis = X_scaled
    x_label, y_label = X_raw.columns[0], X_raw.columns[1]

plt.figure(figsize=(7,5))

# Data points
plt.scatter(
    X_vis[:, 0],
    X_vis[:, 1],
    c=labels,
    s=60
)

# Centroids (project if PCA used)
if X_scaled.shape[1] > 2:
    centroids_vis = pca.transform(centroids)
else:
    centroids_vis = centroids

plt.scatter(
    centroids_vis[:, 0],
    centroids_vis[:, 1],
    marker='X',
    s=200
)

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title("K-Means Clustering with Centroids")
plt.show()


# =========================
# 11. ADD CLUSTER TO DATASET
# =========================
df["Cluster"] = labels
print(df.head())