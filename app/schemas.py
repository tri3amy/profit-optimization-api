from pydantic import BaseModel, Field

class BudgetRequest(BaseModel):
    total_budget: float = Field(..., gt=0, description="Total Available Advertising Budget")

class Allocation(BaseModel):
    tv: float
    radio: float
    newspaper: float

class OptimizationResponse(BaseModel):
    status: str
    total_budget: float
    allocation: Allocation
    predicted_sales: float
    total_spent: float
    remaining_budget: float