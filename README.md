Housing price prediction project: preprocesses numeric/categorical features (log-transform, scaling, one-hot encoding), engineers polynomial and interaction terms, evaluates best-subset regression and regularised linear models versus XGBoost, and selects the final model using holdout-set RMSE and back-transformed metric

## Running the notebook

### Install dependencies (Python 3.12)
```bash
pip install -r requirements.txt
```


### Outputs
- Processed data: `data/processed/`
- Model artefacts: `models/`
- Figures: `reports/figures/`
