# Cài thư viện nếu cần
# !pip3 install pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import random

# Đọc dữ liệu
df = pd.read_csv("D:/Visual Studio Code/KMeans/Kmeans/customer_data.csv")
features = ["age", "income", "purchase_amount", "promotion_usage", "satisfaction_score"]
X = df[features]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X[features] = scaler.fit_transform(X[features])

# PCA để trực quan hóa nếu muốn
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Định nghĩa lớp Point và DBSCAN from scratch
class Point:
    def __init__(self, coordinate):
        self.coordinate = coordinate
        self.cluster_idx = None

    def is_clustered(self):
        return self.cluster_idx is not None

    def cluster(self, cluster_idx):
        self.cluster_idx = cluster_idx

class DBSCAN:
    def __init__(self, epsi, min_points):
        self.epsi = epsi
        self.min_points = min_points

    @staticmethod
    def _compute_distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    def _find_neighbor_indices(self, core_point, other_points):
        return [
            idx for idx, pt in enumerate(other_points)
            if self._compute_distance(core_point.coordinate, pt.coordinate) <= self.epsi
        ]

    def fit(self, df, feature_cols):
        points = [Point(row.to_numpy()) for _, row in df[feature_cols].iterrows()]
        free_point_indices = set(range(len(points)))
        core_point_indices = set()
        cluster_count = 0

        while free_point_indices:
            if not core_point_indices:
                start_idx = random.choice(list(free_point_indices))
                core_point_indices.add(start_idx)
                free_point_indices.remove(start_idx)
                points[start_idx].cluster(cluster_count)

            while core_point_indices:
                picked_core_point_idx = core_point_indices.pop()
                neighbor_indices = self._find_neighbor_indices(points[picked_core_point_idx], points)
                if len(neighbor_indices) >= (self.min_points - 1):
                    for neighbor_idx in neighbor_indices:
                        if neighbor_idx in free_point_indices:
                            points[neighbor_idx].cluster(cluster_count)
                            core_point_indices.add(neighbor_idx)
                            free_point_indices.remove(neighbor_idx)

            cluster_count += 1
        return points

# Tìm eps và min_samples tối ưu
epsilon = [1, 1.5, 2, 2.5, 3, 3.5, 4]
min_samples = [10, 15, 20, 25]
sil_avg = []
dbi_scores = []
max_value = [0, 0, 0, -1]

for eps in epsilon:
    for min_s in min_samples:
        db = DBSCAN(epsi=eps, min_points=min_s).fit(X, feature_cols=features)
        labels = [point.cluster_idx if point.cluster_idx is not None else -1 for point in db]

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_ < 2:
            continue

        filtered_X = X[np.array(labels) != -1]
        filtered_labels = np.array(labels)[np.array(labels) != -1]

        sil_score = silhouette_score(filtered_X, filtered_labels)
        dbi_score = davies_bouldin_score(filtered_X, filtered_labels)

        sil_avg.append((eps, min_s, sil_score))
        dbi_scores.append((eps, min_s, dbi_score))

        if sil_score > max_value[3]:
            max_value = [eps, min_s, n_clusters_, sil_score]

# In kết quả tốt nhất
print("Tốt nhất:")
print("epsilon =", max_value[0])
print("min_samples =", max_value[1])
print("number of clusters =", max_value[2])
print("average silhouette score = %.4f" % max_value[3])

# Vẽ biểu đồ Silhouette Score theo eps và min_samples
fig = plt.figure(figsize=(10, 5))
for eps, min_s, score in sil_avg:
    plt.scatter(eps, score, c='b')
plt.title("Silhouette Score theo epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Vẽ biểu đồ Davies–Bouldin Index
fig = plt.figure(figsize=(10, 5))
for eps, min_s, score in dbi_scores:
    plt.scatter(eps, score, c='r')
plt.title("Davies–Bouldin Index theo epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Davies–Bouldin Index")
plt.grid(True)
plt.show()
