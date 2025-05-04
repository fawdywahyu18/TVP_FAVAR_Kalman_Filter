# TVP_FAVAR_Kalman_Filter

## Overview

This repository contains Python code for estimating and forecasting macroeconomic variables using Time-Varying Parameter Factor-Augmented Vector Autoregression (TVP-FAVAR) with Kalman filtering. The implementation is based on the methodology from Koop and Korobilis (2014) and is designed for analyzing time series data with evolving relationships over time.

## Features

- **Time-Varying Parameters**: Captures evolving relationships between macroeconomic variables
- **Factor Augmentation**: Extracts latent factors from large datasets to reduce dimensionality
- **Kalman Filtering and Smoothing**: Provides optimal state estimation for time-varying parameters
- **Data Simulation**: Includes functionality to generate realistic macroeconomic data for testing
- **Model Validation**: Tools to validate model performance against simulated ground truth
- **Forecasting**: Generates multi-step ahead forecasts with the estimated model

## Required Packages
``` python
numpy
pandas
matplotlib
scipy
scikit-learn
statsmodels
```

## Usage

### Basic Example

```python
import numpy as np
from tvp_favar import run_tvp_favar

# Run TVP-FAVAR with simulated data
np.random.seed(123)
results = run_tvp_favar(nfac=1, nlag=2, validate=True, plot_results=True)

# Access results
estimated_factors = results['estimated_factors']
factor_loadings = results['factor_loadings']
var_coefficients = results['var_coefficients']
forecasts = results['forecasts']
validation = results['validation']
```

## Key Functions

### Simulation and Data Preparation
- `simulate_data()`: Generates synthetic data with realistic macroeconomic properties  
- `standardize_miss()`: Standardizes data while handling missing values  

### Model Estimation
- `extract()`: Extracts factors using Principal Component Analysis  
- `olssvd()`: Performs OLS estimation using Singular Value Decomposition  
- `mlag2()`: Creates matrix of lagged values  
- `Minn_prior_KOOP()`: Implements Minnesota prior for VAR coefficients  

### Kalman Filtering and Smoothing
- `KFS_parameters()`: Performs Kalman filtering and smoothing for parameters  
- `KFS_factors()`: Performs Kalman filtering and smoothing for factors  

### Validation and Forecasting
- `validate_tvp_favar()`: Validates model performance using simulated data  
- `run_tvp_favar()`: Main function to run the TVP-FAVAR model  

## Model Parameters
- `nfac`: Number of factors to extract *(default: 1)*  
- `nlag`: Number of lags in the VAR model *(default: 4)*  
- `decay_factors`: Decay factors for covariance matrices `[l_1, l_2, l_3, l_4]` *(default: [0.96, 0.96, 0.99, 0.99])*  
- `y_true`: Flag indicating if Y should be included in measurement equation *(default: 1)*  

## Validation Metrics
The model provides several validation metrics when used with simulated data:
- **Factor Correlation**: Measures correlation between true and estimated factors  
- **Stationarity**: Augmented Dickey-Fuller test for stationarity of estimated factors  
- **Forecast Metrics**: MAE, RMSE, MAPE, and Directional Accuracy (when applicable)


## References
- Koop, G., & Korobilis, D. (2014). A new index of financial conditions. *European Economic Review*, 71, 101-116.  
- Primiceri, G. E. (2005). Time varying structural vector autoregressions and monetary policy. *The Review of Economic Studies*, 72(3), 821-852.  
- Doz, C., Giannone, D., & Reichlin, L. (2011). A two-step estimator for large approximate dynamic factor models based on Kalman filtering. *Journal of Econometrics*, 164(1), 188-205.  
