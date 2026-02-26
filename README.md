# Housing price prediction 

## Feature Engineering
- Log-transformed skewed numeric variables
- Standardised continuous predictors
- One-hot encoded categorical variables
- Added polynomial & interaction terms

## Hyperparameter Tuning

XGBoost hyperparameters were optimised using a structured **sequential stepwise tuning strategy** to reduce search complexity and improve stability.

The key parameters were grouped and tuned in stages:

- **Group 1:** `max_depth`, `min_child_weight`  
- **Group 2:** `subsample`, `colsample_bytree`  
- **Group 3:** `learning_rate`, `num_boost_round`  

Tuning procedure:
1. Initialise `learning_rate = 0.1` and `num_boost_round = 1000`.
2. Tune **Group 1** via cross-validated RMSE.
3. Fix Group 1 at optimal values and tune **Group 2**.
4. Fix Groups 1–2 and tune **Group 3** last.

At each stage, previously tuned parameters were fixed while remaining parameters stayed at default values.  

This staged optimisation reduces the dimensionality of the hyperparameter space, improving computational efficiency while maintaining model performance.

## Final Model Selection

Selected XGBoost based on lowest RMSE

|Model       |RMSE             |MAE              |R2                |MAPE              |
|------------|-----------------|-----------------|------------------|------------------|
|XGBoost     |380954.7045424975|194292.2833333335|0.9357303329398784|7.000255854793516 |
|RidgeCV     |555101.8378534262|299727.3985739661|0.8635402101418502|10.167425604333639|
|LassoCV     |560069.8841926266|307249.3822219568|0.8610867062339951|10.53069130403988 |
|ElasticNetCV|560321.7717420819|307348.1894182262|0.8609617275536008|10.532968343838824|


<img src="reports/figures/holdout_actual_vs_pred.png" width="600">

<img src="reports/figures/holdout_residual_hist.png" width="600">

## Running the notebook

### Install dependencies (Python 3.12)
```bash
pip install -r requirements.txt
```


### Outputs
- Processed data: `data/processed/`
- Model artefacts: `models/`
- Figures: `reports/figures/`
