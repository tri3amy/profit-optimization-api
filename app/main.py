from fastapi import FastAPI, HTTPException
from app.schemas import BudgetRequest, OptimizationResponse
from app.optimizer import run_optimization
import numpy as np

app = FastAPI(
    title="Profit Optimization API",
    description="API for recommending ad budget allocation using Machine Learning.",
    version="1.0.0"
)

@app.post("/optimize/", response_model=OptimizationResponse)
def optimize_budget(request: BudgetRequest):
    result = run_optimization(request.total_budget)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to optimize.")
    
    best_x = result["best_x"]
    best_sales = result["best_sales"]
    total_spent = np.sum(best_x)
    
    return {
        "status": "Success",
        "total_budget": request.total_budget,
        "allocation": {
            "tv": round(best_x[0], 2),
            "radio": round(best_x[1], 2),
            "newspaper": round(best_x[2], 2)
        },
        "predicted_sales": round(best_sales, 2),
        "total_spent": round(total_spent, 2),
        "remaining_budget": round(request.total_budget - total_spent, 2)
    }