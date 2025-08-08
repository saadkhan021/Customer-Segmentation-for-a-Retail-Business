# customer_segmentation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\saadk\Desktop\Week3 intern\Mall_Customers.csv")

# Step 2: Basic EDA
print("First 5 rows:")
print(df.head())
print("\nSummary:")
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Optional: Encode Gender if needed
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Step 3: Select features for clustering
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Elbow Method to choose k
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot elbow
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

# Step 5: Fit final KMeans model with optimal k (e.g., 5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 6: PCA for 2D visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

# Step 7: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=80)
plt.title('Customer Segments Visualized with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Step 8: Analyze clusters
for i in range(optimal_k):
    print(f"\nCluster {i}:")
    print(df[df['Cluster'] == i][['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())



# Evaluate clustering quality
sil_score = silhouette_score(scaled_features, df['Cluster'])
db_score = davies_bouldin_score(scaled_features, df['Cluster'])

print(f"\n Clustering Evaluation Metrics:")
print(f"Silhouette Score: {sil_score:.3f} (higher is better)")
print(f"Davies-Bouldin Index: {db_score:.3f} (lower is better)")




