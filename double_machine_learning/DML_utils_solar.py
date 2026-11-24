import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from analysis_settings import UK_COLUMNS
from data_pipeline import schema


def residualize_variable_kfold(data, y_var, x_vars, n_folds=5):
    X = data[x_vars].values
    y = data[y_var].values
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    residuals = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.1,
            'max_depth': 5,
            'num_threads': 1,  # Single thread is much faster on small data (and lets us parallelise windows)
            'verbose': -1
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        model = lgb.train(params, lgb_train, num_boost_round=100)

        # Predict and calculate residuals
        y_pred = model.predict(X_test)
        residuals[test_idx] = y_test - y_pred

    return residuals


# Residualize data
# cols maps standard names (schema.py) to the dataset's column names; default is original UK names
def residualize_data(data, y=None, n_folds=5, cols=UK_COLUMNS):
    c = cols
    y = y or c[schema.PRICE]
    data['solar_production_resid'] = residualize_variable_kfold(
        data, 
        c[schema.SOLAR_FORECAST], 
        [c[schema.SOLAR_CAPACITY], c[schema.DAYLIGHT_HOURS], c[schema.HOUR], c[schema.MONTH]], 
        n_folds=n_folds
    )
    confounders = [c[schema.DAYLIGHT_HOURS], c[schema.HOUR], c[schema.YEAR], c[schema.MONTH], c[schema.DAY_OF_WEEK],
                   c[schema.LOAD_FORECAST], c[schema.GAS_PRICE], c[schema.CO2_PRICE], c[schema.WIND_FORECAST]]
    data['spot_price_resid'] = residualize_variable_kfold(
        data, y, confounders, n_folds=n_folds
    )
    return data


# Fit residualized model
def fit_residualized_model(data):
    scaler = StandardScaler()
    X_residualized = data[['solar_production_resid']]
    y_residualized = data['spot_price_resid']

    pipeline_residualized = Pipeline([
        ('scaler', scaler),
        ('regressor', LinearRegression())
    ])

    pipeline_residualized.fit(X_residualized, y_residualized)
    coef_residualized = pipeline_residualized.named_steps['regressor'].coef_
    scale_residualized = pipeline_residualized.named_steps['scaler'].scale_

    return coef_residualized / scale_residualized
