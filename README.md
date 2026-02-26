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

'| Model        |   RMSE |    MAE |       R2 |     MAPE |\n|:-------------|-------:|-------:|---------:|---------:|\n| XGBoost      | 380955 | 194292 | 0.93573  |  7.00026 |\n| RidgeCV      | 555102 | 299727 | 0.86354  | 10.1674  |\n| LassoCV      | 560070 | 307249 | 0.861087 | 10.5307  |\n| ElasticNetCV | 560322 | 307348 | 0.860962 | 10.533   |'

Selected XGBoost based on lowest RMSE

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
