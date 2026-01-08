# ================================
# HIERARCHICAL CLUSTERING (GENERAL)
# ================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------
# 1. LOAD DATASET
# ----------------
df = pd.read_csv("/Users/samarthgarg/Downloads/ML ETP/DataSets/Mall_Customers.csv")   # 🔁 CHANGE PATH
print(df.head())
print(df.info())

# -------------------------
# 2. SELECT NUMERIC FEATURES
# -------------------------
features = ["Annual Income (k$)", "Spending Score (1-100)"]  # 🔁 CHANGE FEATURES
X = df[features]

# ----------------
# 3. STANDARDIZE DATA
# ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------
# 4. DENDROGRAM (CLUSTER DECISION)
# ----------------
plt.figure(figsize=(15,7))
linked = linkage(X_scaled, method='ward')
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# ----------------
# 5. APPLY AGGLOMERATIVE CLUSTERING
# ----------------
k = 3   # 🔁 CHOOSE FROM DENDROGRAM

hc = AgglomerativeClustering(
    n_clusters=k,
    linkage='ward'
)

clusters = hc.fit_predict(X_scaled)
df["Cluster"] = clusters

# ----------------
# 6. PCA FOR VISUALIZATION
# ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ----------------
# 7. VISUALIZE CLUSTERS
# ----------------
plt.figure(figsize=(8,6))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap='tab10',
    s=60
)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Hierarchical Clustering Visualization")
plt.show()

# ----------------
# 8. MODEL EVALUATION
# ----------------
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

# ----------------
# 9. FINAL OUTPUT
# ----------------
print("\nClustered Data Sample:")
print(df.head())
