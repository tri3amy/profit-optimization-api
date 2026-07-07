import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'pipeline_lasso_poly.pkl')

pipeline_log = joblib.load(MODEL_PATH)

MAX_TV, MAX_RADIO, MAX_NEWS = 293.6, 49.6, 89.4

def predict_sales(spends):
    tv, radio, news = spends
    news_log = np.log1p(news)
    input_df = pd.DataFrame([[tv, radio, news_log]],
                            columns=['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper_log'])
    log_pred = pipeline_log.predict(input_df)[0]
    return np.expm1(log_pred)

def run_optimization(total_budget: float):
    bounds = [(0, MAX_TV), (0, MAX_RADIO), (0, MAX_NEWS)]
    cons = {'type': 'ineq', 'fun': lambda x: total_budget - np.sum(x)}
    
    guess = [min(total_budget/2, MAX_TV), min(total_budget/4, MAX_RADIO), min(total_budget/4, MAX_NEWS)]
    guess = np.clip(guess, [0, 0, 0], [MAX_TV, MAX_RADIO, MAX_NEWS])
    
    res = minimize(lambda x: -predict_sales(x), guess, bounds=bounds, constraints=cons, method='SLSQP')
    
    if res.success:
        return {"best_x": res.x, "best_sales": -res.fun}
    return None