from django.shortcuts import render
import joblib
from model_generators.clustering.train_cluster import evaluate_clustering_model 
from model_generators.classification.train_classifier import evaluate_classification_model 
from model_generators.regression.train_regression import evaluate_regression_model 
 
# Load models once 
regression_model = joblib.load("model_generators/regression/regression_model.pkl") 
classification_model = joblib.load("model_generators/classification/classification_model.pkl") 
clustering_model = joblib.load("model_generators/clustering/clustering_model.pkl")
clustering_scaler = joblib.load("model_generators/clustering/clustering_scaler.pkl") 
 
def classification_analysis(request): 
    context = { 
        "evaluations": evaluate_classification_model() 
    } 
    if request.method == "POST": 
        year = int(request.POST["year"]) 
        km = float(request.POST["km"]) 
        seats = int(request.POST["seats"]) 
        income = float(request.POST["income"]) 
        prediction = classification_model.predict([[year, km, seats, income]])[0] 
 
        context["prediction"] = prediction 
    return render(request, "predictor/classification_analysis.html", context) 
 
def clustering_analysis(request): 
    context = { 
        "evaluations": evaluate_clustering_model() 
    } 
    if request.method == "POST": 
        try: 
            year = int(request.POST["year"]) 
            km = float(request.POST["km"]) 
            seats = int(request.POST["seats"]) 
            income = float(request.POST["income"]) 
            # Step 1: Predict price 
            predicted_price = regression_model.predict([[year, km, seats, income]])[0] 
            # Step 2: Scale features and predict cluster
            features_to_cluster = [[income, predicted_price]]
            features_scaled = clustering_scaler.transform(features_to_cluster)
            cluster_id = clustering_model.predict(features_scaled)[0] 
            mapping = { 
                0: "Economy", 
                1: "Standard", 
                2: "Premium" 
            } 
            context.update({ 
                "prediction": mapping.get(cluster_id, "Unknown"), 
                "price": round(predicted_price, 2)
            }) 
        except Exception as e: 
            context["error"] = str(e) 
 
    return render(request, "predictor/clustering_analysis.html", context) 

import pandas as pd
from predictor.data_eploration import dataset_exploration
from predictor.map_visualization import create_rwanda_district_map, get_district_summary_table

def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    context = {
        "table": dataset_exploration(df),
        "map": create_rwanda_district_map(df),
        "district_summary": get_district_summary_table(df),
        "total_clients": len(df),
        "total_districts": df['district'].nunique(),
        "total_provinces": df['province'].nunique()
    }
    return render(request, "predictor/index.html", context)

def regression_analysis(request):
    context = {
        "evaluations": evaluate_regression_model()
    }
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context["prediction"] = round(prediction, 2)
    return render(request, "predictor/regretion_analysis.html", context)
