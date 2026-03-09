import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

print("Testing different feature combinations and preprocessing methods...\n")

best_score = 0
best_config = None

# Test different feature combinations
feature_sets = [
    ["estimated_income", "selling_price"],
    ["estimated_income", "selling_price", "year"],
    ["estimated_income", "selling_price", "kilometers_driven"],
    ["estimated_income", "selling_price", "year", "kilometers_driven"],
]

scalers = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler()
}

for features in feature_sets:
    X = df[features]
    
    for scaler_name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        
        # Try different numbers of clusters
        for n_clusters in [2, 3, 4, 5]:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=100,
                max_iter=1000,
                algorithm='lloyd'
            )
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_config = {
                    'features': features,
                    'scaler': scaler_name,
                    'n_clusters': n_clusters,
                    'score': score
                }
            
            print(f"Features: {features}, Scaler: {scaler_name}, Clusters: {n_clusters}, Score: {score:.4f}")

print(f"\n{'='*80}")
print(f"BEST CONFIGURATION:")
print(f"Features: {best_config['features']}")
print(f"Scaler: {best_config['scaler']}")
print(f"Number of Clusters: {best_config['n_clusters']}")
print(f"Silhouette Score: {best_config['score']:.4f}")
print(f"{'='*80}")
