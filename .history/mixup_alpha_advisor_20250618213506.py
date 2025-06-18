import joblib
import numpy as np
from fastapi import APIRouter, Request

router = APIRouter()

# Load the model once at startup
model = joblib.load('mixup_alpha_advisor.joblib')

@router.post("/api/recommend_mixup_alpha")
async def recommend_mixup_alpha(request: Request):
    config = await request.json()
    features = np.array([config[feat] for feat in model.feature_names_in_]).reshape(1, -1)
    pred = model.predict(features)[0]
    return {"predicted_accuracy": float(pred)}

@router.post("/api/sweep_mixup_alpha")
async def sweep_mixup_alpha(request: Request):
    config = await request.json()
    candidate_alphas = np.arange(0.0, 2.1, 0.1)
    best_alpha = None
    best_pred = -np.inf
    for alpha in candidate_alphas:
        config['Mixup_Alpha'] = alpha
        features = np.array([config[feat] for feat in model.feature_names_in_]).reshape(1, -1)
        pred = model.predict(features)[0]
        if pred > best_pred:
            best_pred = pred
            best_alpha = alpha
    return {"recommended_mixup_alpha": best_alpha, "predicted_accuracy": float(best_pred)} 