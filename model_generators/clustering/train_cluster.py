import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
from sklearn.preprocessing import StandardScaler
import joblib 
 
 
SEGMENT_FEATURES = ["estimated_income", "selling_price"] 
df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv") 
X = df[SEGMENT_FEATURES] 

# Standardize features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Improved clustering with more iterations and better initialization
kmeans = KMeans(n_clusters=3, random_state=42, n_init=50, max_iter=500, algorithm='lloyd') 
df["cluster_id"] = kmeans.fit_predict(X_scaled) 
centers = kmeans.cluster_centers_ 

# Transform centers back to original scale for interpretation
centers_original = scaler.inverse_transform(centers)

# Sort clusters by income 
sorted_clusters = centers_original[:, 0].argsort() 
 
cluster_mapping = { 
    sorted_clusters[0]: "Economy", 
    sorted_clusters[1]: "Standard", 
    sorted_clusters[2]: "Premium", 
} 
 
df["client_class"] = df["cluster_id"].map(cluster_mapping) 
 
# Save both model and scaler
joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl") 
joblib.dump(scaler, "model_generators/clustering/clustering_scaler.pkl")

silhouette_avg = round(silhouette_score(X_scaled, df["cluster_id"]), 4)

# Calculate Coefficient of Variation for each cluster
cluster_stats = []
for cluster_name in ["Economy", "Standard", "Premium"]:
    cluster_data = df[df["client_class"] == cluster_name][SEGMENT_FEATURES]
    mean_vals = cluster_data.mean()
    std_vals = cluster_data.std()
    cv_vals = (std_vals / mean_vals) * 100
    cluster_stats.append({
        'cluster': cluster_name,
        'income_cv': round(cv_vals['estimated_income'], 2),
        'price_cv': round(cv_vals['selling_price'], 2),
        'avg_cv': round(cv_vals.mean(), 2)
    })

coefficient_of_variation = round(np.mean([stat['avg_cv'] for stat in cluster_stats]), 2) 
 
cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean() 
cluster_counts = df["client_class"].value_counts().reset_index() 
cluster_counts.columns = ["client_class", "count"] 
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class") 
comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]] 
 
 
 
def evaluate_clustering_model(): 
    # Create CV stats table
    cv_stats_df = pd.DataFrame(cluster_stats)
    cv_table = cv_stats_df.to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False
    )
    
    return { 
        "silhouette": silhouette_avg,
        "coefficient_of_variation": coefficient_of_variation,
        "cv_details": cv_table,
        "summary": cluster_summary.to_html( 
            classes="table table-bordered table-striped table-sm", 
            float_format="%.2f", 
            justify="center", 
            index=False, 
        ), 
        "comparison": comparison_df.head(10).to_html( 
            classes="table table-bordered table-striped table-sm", 
            float_format="%.2f", 
            justify="center", 
            index=False, 
        ), 
    } 