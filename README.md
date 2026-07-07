# Profit Optimization & Advertising Budget Allocation API
**Prescriptive Analytics: Maximizing Sales using Polynomial Lasso Regression, SciPy Optimization, and Dockerized FastAPI**

![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![R2 Score](https://img.shields.io/badge/R%C2%B2_Score-97.30%25-blue)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Deployment](https://img.shields.io/badge/Deployment-Docker-2496ED)

Python, Scikit-Learn, SciPy, Pandas, FastAPI, Uvicorn, Docker, MLOps.

## Repository Structure
```text
profit-optimization-advertising/
│
├── app/
│   ├── __init__.py                         # Package initializer
│   ├── main.py                             # FastAPI application endpoints
│   ├── optimizer.py                        # SciPy optimization engine
│   └── schemas.py                          # Pydantic data validation models
│
├── data/
│   └── Advertising Budget and Sales.csv    # Raw dataset
│
├── model/
│   └── pipeline_lasso_poly.pkl             # Saved Machine Learning Pipeline (Trained)
│
├── notebook/
│   └── part.ipynb                          # Main Jupyter Notebook (EDA & Model Training)
│
├── .gitignore                              # Git ignore file
├── Dockerfile                              # Docker container instructions
├── environment.yml                         # Conda environment configuration (Historical)
├── LICENSE                                 # Project license
├── README.md                               # Project documentation
└── requirements.txt                        # API and model dependencies
```

## Project Overview
* **Problem Statement:** Marketing teams often distribute advertising budgets evenly across channels or rely solely on gut feeling. The challenge is: "How do we allocate a limited advertising budget (TV, Radio, Newspaper) to maximize Sales?"
* **Objective:** This project aims to transition from merely predicting (Predictive Analytics) to recommending actionable business decisions (Prescriptive Analytics). The system identifies the saturation point of advertising effectiveness and formulates a budget portfolio that yields the highest Return on Investment (ROI). The solution is packaged as a robust API for seamless integration with frontend or dashboard applications.
* **Key Result:** The regression model achieved an Accuracy (R² Score) of 97.30% with an RMSE of 0.852. The optimization algorithm mathematically proved that the budget does not need to be fully spent if it has reached Diminishing Marginal Returns.

## Key Business Insights
Based on exploratory data analysis and optimization results, several crucial insights were discovered:
* **The TV & Radio Synergy (Bow-tie Pattern):** The model identified a strong two-way interaction effect. TV and Radio advertising deliver a massive surge in sales only when executed simultaneously, rather than individually.
* **Newspaper is a Dead Weight:** Newspaper advertising proved to have a very weak correlation with sales and poor marginal efficiency. The optimization algorithm consistently slashed the newspaper budget to $0 and reallocated it to TV/Radio.
* **The Saturation Point:** The Law of Diminishing Returns is highly evident in this dataset. Expanding the total budget beyond $350 yields near-zero additional Sales (Marginal Return < 0.005). Surplus funds are better saved or allocated to other departments.

## Methodology & Architecture

### 1. Machine Learning Pipeline
* **Log Transformation (`np.log1p`):** Applied to the Newspaper variable and target Sales to cure right-skewness and reduce heteroscedasticity.
* **StandardScaler & Polynomial Features (Degree=2):** Equalized feature scales and generated quadratic features to capture the Diminishing Returns curvature and TV x Radio synergy.
* **LassoCV (L1 Regularization):** Acted as a strict Feature Selection mechanism, eliminating useless polynomial features.
* **SciPy Minimize (SLSQP):** An optimization algorithm used to find the input budget combination that produces the highest output (Sales) without exceeding the maximum budget constraint.

### 2. MLOps & API Architecture
* **FastAPI:** Built a high-performance REST API to serve the SciPy optimization engine.
* **Pydantic:** Ensured strict data validation for incoming budget requests.
* **Docker:** Containerized the entire application (OS, Python, dependencies, and model) into a lightweight image to eliminate the "it works on my machine" problem, making it 100% production-ready.

## Installation & Deployment (Docker)
The easiest and most reliable way to run this application is using Docker. Ensure you have Docker Desktop installed and running on your machine.

**1. Clone this repository:**
```bash
git clone [https://github.com/tri3amy/profit-optimization-advertising.git](https://github.com/tri3amy/profit-optimization-advertising.git)
cd profit-optimization-advertising
```

**2. Build the Docker Image:**
```bash
docker build -t profit-api .
```

**3. Run the Docker Container:**
```bash
docker run -d -p 8000:8000 profit-api
```

## Usage (Interactive Swagger UI)
Once the Docker container is running, the API provides an out-of-the-box interactive interface.

1. Open your web browser and navigate to: **http://localhost:8000/docs**
2. Expand the `POST /optimize/` endpoint and click **"Try it out"**.
3. Enter your desired total budget in the request body, for example:
```json
{
  "total_budget": 350
}
```
4. Click **Execute** to receive the AI-recommended budget allocation in real-time.

**Example API Response:**
```json
{
  "status": "Success",
  "total_budget": 350.0,
  "allocation": {
    "tv": 293.6,
    "radio": 49.6,
    "newspaper": 0.0
  },
  "predicted_sales": 30.06,
  "total_spent": 343.2,
  "remaining_budget": 6.8
}
```

## Future Improvements
* **Time-Series / Seasonality:** Incorporating time elements as advertising effectiveness can be highly dependent on seasonality.
* **Cost of Goods Sold (COGS) Integration:** Shifting the optimization target from "Maximum Revenue" to "Maximum Net Profit" by incorporating production cost data.
* **CI/CD Pipeline:** Implementing GitHub Actions to automatically trigger Docker builds and tests whenever new code or a retrained model is pushed to the repository.
