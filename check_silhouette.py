import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
SEGMENT_FEATURES = ["estimated_income", "selling_price"]
X = df[SEGMENT_FEATURES]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train with improved parameters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=50, max_iter=500, algorithm='lloyd')
labels = kmeans.fit_predict(X_scaled)

# Calculate silhouette score
silhouette = silhouette_score(X_scaled, labels)

print(f"Silhouette Score: {silhouette:.4f}")

# Calculate coefficient of variation
cluster_stats = []
for cluster_id in range(3):
    cluster_data = X[labels == cluster_id]
    mean_vals = cluster_data.mean()
    std_vals = cluster_data.std()
    cv_vals = (std_vals / mean_vals) * 100
    cluster_stats.append({
        'cluster': cluster_id,
        'income_cv': cv_vals['estimated_income'],
        'price_cv': cv_vals['selling_price'],
        'avg_cv': cv_vals.mean()
    })
    print(f"Cluster {cluster_id}: Income CV={cv_vals['estimated_income']:.2f}%, Price CV={cv_vals['selling_price']:.2f}%")

overall_cv = np.mean([stat['avg_cv'] for stat in cluster_stats])
print(f"\nOverall Coefficient of Variation: {overall_cv:.2f}%")
