#Problem Statement: To find the types of gym users in the gym using Unsupervised Learning models and compare them to find the most
#efficient model


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("gym_membership.csv")

print(df.shape)
df.head()

# create a copy of dataframe and

data = df.copy()

data["attend_group_lesson"] = data["attend_group_lesson"].astype(int)
data["drink_abo"] = data["drink_abo"].astype(int)
data["personal_training"] = data["personal_training"].astype(int)
data["uses_sauna"] = data["uses_sauna"].astype(int)

features = data[
    [
        "Age",
        "visit_per_week",
        "avg_time_in_gym",
        "attend_group_lesson",
        "drink_abo",
        "personal_training",
        "uses_sauna"
    ]
]

features.head()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)

print(X_scaled.shape)

wcss = []

for k in range(1,11):
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

data["KMeans_Cluster"] = kmeans_labels

sil_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

db_score = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)

print("KMeans Results")
print("Silhouette Score:", round(sil_score,4))
print("Davies Bouldin Score:", round(db_score,4))

plt.figure(figsize=(12,6))

linked = linkage(
    X_scaled,
    method="ward"
)

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Members")
plt.ylabel("Distance")

plt.show()

agg = AgglomerativeClustering(
    n_clusters=4
)

agg_labels = agg.fit_predict(X_scaled)

data["Hierarchical_Cluster"] = agg_labels

print(
    "Hierarchical Silhouette:",
    silhouette_score(X_scaled, agg_labels)
)

print(
    "Hierarchical DB Index:",
    davies_bouldin_score(X_scaled, agg_labels)
)

dbscan = DBSCAN(
    eps=1.2,
    min_samples=8
)

dbscan_labels = dbscan.fit_predict(X_scaled)

data["DBSCAN_Cluster"] = dbscan_labels

unique_clusters = len(set(dbscan_labels))

print("Clusters found:", unique_clusters)

mask = dbscan_labels != -1

if len(set(dbscan_labels[mask])) > 1:

    print(
        "DBSCAN Silhouette:",
        silhouette_score(
            X_scaled[mask],
            dbscan_labels[mask]
        )
    )

    print(
        "DBSCAN DB Index:",
        davies_bouldin_score(
            X_scaled[mask],
            dbscan_labels[mask]
        )
    )

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1","PC2"]
)

pca_df["Cluster"] = kmeans_labels

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set1"
)

plt.title("KMeans Clusters (PCA 2D)")
plt.show()

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

data["Anomaly"] = iso.fit_predict(X_scaled)

data["Anomaly"] = data["Anomaly"].map(
    {
        1:"Normal",
        -1:"Anomaly"
    }
)

print(
    data["Anomaly"].value_counts()
)

pca_df["Anomaly"] = data["Anomaly"]

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Anomaly"
)

plt.title("Isolation Forest Anomaly Detection")
plt.show()

results = pd.DataFrame({

    "Algorithm":[
        "KMeans",
        "Hierarchical"
    ],

    "Silhouette":[

        silhouette_score(
            X_scaled,
            kmeans_labels
        ),

        silhouette_score(
            X_scaled,
            agg_labels
        )
    ],

    "Davies_Bouldin":[

        davies_bouldin_score(
            X_scaled,
            kmeans_labels
        ),

        davies_bouldin_score(
            X_scaled,
            agg_labels
        )
    ]
})

print(results)