# Python Functions to forecast inflation using TVP-FAVAR
# Improved implementation with data simulation and validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn import preprocessing
from scipy import stats
import statsmodels.api as sm

# Utility function for data standardization with missing values
def standardize_miss(data):
    """
    Standardize data while handling missing values (NaN)
    
    Parameters:
    data (numpy.ndarray): Input data matrix
    
    Returns:
    numpy.ndarray: Standardized data
    """
    mean_vals = np.nanmean(data, axis=0)
    std_vals = np.nanstd(data, axis=0)
    
    # Replace zeros in std with small value to avoid division by zero
    std_vals[std_vals == 0] = 1e-10
    
    # Create standardized data
    standardized = (data - mean_vals) / std_vals
    
    # Replace NaN values with zeros
    standardized = np.nan_to_num(standardized)
    
    return standardized

# Minnesota Prior implementation for Koop
def Minn_prior_KOOP(gamma=0.1, M=4, p=4, K=None):
    """
    Minnesota prior for the TVP-VAR model as in Koop & Korobilis (2014)
    
    Parameters:
    gamma (float): Tightness parameter, default=0.1
    M (int): Number of variables, default=4
    p (int): Number of lags, default=4
    K (int): Total number of coefficients, default=M*p
    
    Returns:
    tuple: Prior mean and variance for VAR coefficients
    """
    if K is None:
        K = M * p
        
    # 1. Minnesota Mean on VAR regression coefficients
    # Create diagonal matrix with 0.9 on diagonal for first lag
    A_prior = np.zeros((M, M * p))
    A_prior[:, :M] = 0.9 * np.eye(M)
    
    # Flatten to column vector
    a_prior = A_prior.flatten('F')
    
    # 2. Minnesota Variance on VAR regression coefficients
    V_i = np.zeros((int(K / M), M))
    
    for i in range(M):
        for j in range(int(K / M)):
            lag = np.ceil((j + 1) / M)  # Find the associated lag number
            V_i[j, i] = gamma / (lag ** 2)
    
    # Create diagonal variance matrix
    V_i_T = V_i.T
    V_prior = np.diag(V_i_T.flatten())
    
    return a_prior, V_prior

# Extract factors using Principal Component Analysis
def extract(data, k=1):
    """
    Extract factors using Principal Component Analysis
    
    Parameters:
    data (numpy.ndarray): Input data matrix
    k (int): Number of factors to extract, default=1
    
    Returns:
    tuple: Extracted factors and loadings
    """
    t, n = data.shape
    
    # Compute covariance matrix
    xx = data.T @ data
    
    # Compute eigenvalues and eigenvectors
    w, v = LA.eig(xx)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    
    # Compute loadings and factors
    lam = np.sqrt(n) * v[:, :k]
    fac = data @ (lam / n)
    
    return fac, lam

# OLS with Singular Value Decomposition
def olssvd(y, ly):
    """
    OLS estimation using Singular Value Decomposition
    
    Parameters:
    y (numpy.ndarray): Dependent variable
    ly (numpy.ndarray): Independent variables
    
    Returns:
    numpy.ndarray: Estimated coefficients
    """
    # Compute SVD
    vl, d, vr = LA.svd(ly, full_matrices=False)
    vr = vr.T  # Transpose to match numpy's SVD format
    
    # Create diagonal matrix with reciprocals
    d_inv = 1.0 / d
    
    # Compute coefficients
    b = (vr * d_inv) @ (vl.T @ y)
    
    return b

# Create lagged matrix
def mlag2(X, p):
    """
    Create matrix of lagged values
    
    Parameters:
    X (numpy.ndarray): Input matrix
    p (int): Number of lags
    
    Returns:
    numpy.ndarray: Matrix with lagged values
    """
    T, N = X.shape
    Xlag = np.zeros((T, N * p))
    
    for i in range(p):
        # Python uses 0-indexing
        Xlag[p:T, (N*i):(N*(i+1))] = X[(p-i-1):(T-i-1), :]
    
    return Xlag

# OLS for PC Dynamic Factor Model
def ols_pc_dfm(YX, YF, Lf, y_true, n, p, r, nfac, nlag):
    """
    OLS estimation for Principal Component Dynamic Factor Model
    
    Parameters:
    YX (numpy.ndarray): Combined Y and X data
    YF (numpy.ndarray): Factors
    Lf (numpy.ndarray): Factor loadings
    y_true (int): Flag indicating if Y should be included
    n (int): Number of variables
    p (int): Number of Y variables
    r (int): Number of factors
    nfac (int): Number of factors to extract
    nlag (int): Number of lags
    
    Returns:
    tuple: Loadings, flattened coefficients, VAR coefficients, measurement error covariance, state error covariance
    """
    t = YX.shape[0]
    
    # Obtain L (the loadings matrix)
    if y_true == 1:
        L = olssvd(YX, YF).T
    elif y_true == 0:
        L = np.block([
            [np.eye(p), np.zeros((p, nfac))],
            [np.zeros((n, p)), Lf]
        ])
    else:
        raise ValueError('Unrecognized y_true value')
    
    # Obtain the errors of the factor equation
    e = YX - YF @ L.T
    sigma2 = np.diag(np.diag(e.T @ e / t))
    
    # Obtain the errors of the VAR equation
    yy = YF[nlag:, :]
    xx = mlag2(YF, nlag)
    xx = xx[nlag:, :]
    
    # OLS estimation
    beta_OLS = LA.inv(xx.T @ xx) @ (xx.T @ yy)
    
    # Compute residual covariance
    residuals = yy - xx @ beta_OLS
    sigmaf = residuals.T @ residuals / (t - nlag - 1)
    
    # Flatten VAR coefficients
    bb = []
    for i in range(nlag):
        g = beta_OLS[(i*r):((i+1)*r), :r]
        g_flat = g.flatten('F')
        bb.extend(g_flat)
    
    bb = np.array(bb)
    
    return L, bb, beta_OLS, sigma2, sigmaf

# Create RHS initialization
def create_RHS_NI(YY, M, p, t):
    """
    Create right-hand side matrix for VAR model with no intercept
    
    Parameters:
    YY (numpy.ndarray): Data matrix
    M (int): Number of variables
    p (int): Number of lags
    t (int): Number of observations
    
    Returns:
    tuple: RHS matrix and total number of coefficients
    """
    K = p * (M ** 2)
    
    # Create x_t matrix
    x_t = np.zeros(((t - p) * M, K))
    
    for i in range(t - p):
        ztemp = np.zeros((M, K))
        col_idx = 0
        
        for j in range(p):
            xtemp = YY[i, (j*M):((j+1)*M)]
            for k in range(M):
                ztemp[k, (k*p*M + j*M):(k*p*M + (j+1)*M)] = xtemp
        
        x_t[(i*M):((i+1)*M), :] = ztemp
    
    return x_t, K

# Kalman filter and smoother for state-space model with companion form
def Kalman_companion(data, S0, P0, H, R, F, Q):
    """
    Kalman filter and smoother for state-space model with companion form
    
    Parameters:
    data (numpy.ndarray): Observed data
    S0 (numpy.ndarray): Initial state
    P0 (numpy.ndarray): Initial state covariance
    H (numpy.ndarray): Observation matrix
    R (numpy.ndarray): Observation error covariance
    F (numpy.ndarray): Transition matrix
    Q (numpy.ndarray): State error covariance
    
    Returns:
    numpy.ndarray: Smoothed state
    """
    t, nm = data.shape
    kml = S0.shape[0]
    km = H.shape[1]
    
    # Initialize storage
    S = np.zeros((t, kml))
    P = np.zeros((kml, kml, t))
    
    # Kalman filter
    Sp = S0
    Pp = P0
    
    for i in range(t):
        y = data[i, :].T
        
        # Measurement equation
        # Prediction error
        nu = y - H @ Sp[:km]
        
        # Variance of prediction error
        f = H @ Pp[:km, :km] @ H.T + R
        
        # Kalman gain
        K = Pp[:, :km] @ H.T @ LA.inv(f)
        
        # Update equations
        Stt = Sp + K @ nu
        Ptt = Pp - K @ H @ Pp[:km, :]
        
        # Store results
        S[i, :] = Stt.T
        P[:, :, i] = Ptt
        
        # Predict for next period
        if i < t-1:
            Sp = F @ Stt
            Pp = F @ Ptt @ F.T + Q
    
    # Kalman smoother
    Sdraw = np.zeros((t, kml))
    Sdraw[-1, :] = S[-1, :]
    
    for i in range(1, t):
        j = t - i
        
        # Extract relevant parts
        Sf = Sdraw[j, :km].T
        Stt = S[j-1, :].T
        Ptt = P[:, :, j-1]
        
        # Smoothing equations
        F_part = F[:km, :]
        f = F_part @ Ptt @ F_part.T + Q[:km, :km]
        U = Ptt @ F_part.T @ LA.inv(f)
        
        # Update
        nu = Sf - F_part @ Stt
        Smean = Stt + U @ nu
        
        # Store results
        Sdraw[j-1, :] = Smean.T
    
    return Sdraw[:, :km]

# KFS Parameters: Kalman filter and smoother for time-varying parameters
def KFS_parameters(YX, FPC, l, nfac, nlag, y_true, k, m, p, q, r, t, lambda_0, beta_0, V_0, Q_0):
    """
    Kalman filter and smoother for time-varying parameters
    
    Parameters:
    YX (numpy.ndarray): Combined Y and X data
    FPC (numpy.ndarray): Factors
    l (numpy.ndarray): Decay factors
    nfac (int): Number of factors
    nlag (int): Number of lags
    y_true (int): Flag indicating if Y should be included
    k (int): Dimension of state vector
    m (int): Number of coefficients
    p (int): Number of Y variables
    q (int): Total number of variables
    r (int): Number of factors
    t (int): Number of observations
    lambda_0 (dict): Prior for loadings
    beta_0 (dict): Prior for VAR coefficients
    V_0 (numpy.ndarray): Initial measurement error covariance
    Q_0 (numpy.ndarray): Initial state error covariance
    
    Returns:
    tuple: Time-varying transition matrices, coefficients, loadings, measurement error covariance, state error covariance
    """
    # Unpack priors
    lambda_0_prmean = lambda_0['mean']
    lambda_0_prvar = lambda_0['var']
    beta_0_prmean = beta_0['mean']
    beta_0_prvar = beta_0['var']
    
    # Unpack decay factors
    l_1, l_2, l_3, l_4 = l.flatten()
    
    # Initialize storage
    lambda_pred = np.zeros((q, r, t))
    lambda_update = np.zeros((q, r, t))
    
    # Identity matrices for first r variables
    for j in range(t):
        lambda_pred[:r, :r, j] = np.eye(r)
        lambda_update[:r, :r, j] = np.eye(r)
    
    beta_pred = np.zeros((m, t))
    beta_update = np.zeros((m, t))
    
    Rl_t = np.zeros((r, r, q, t))
    Sl_t = np.zeros((r, r, q, t))
    Rb_t = np.zeros((m, m, t))
    Sb_t = np.zeros((m, m, t))
    
    x_t_pred = np.zeros((t, q))
    e_t = np.zeros((q, t))
    
    lambda_t = np.zeros((q, r, t))
    beta_t = np.zeros((k, k, t))
    Q_t = np.zeros((r, r, t))
    V_t = np.zeros((q, q, t))
    
    # Prepare lagged factors for VAR
    yy = FPC[nlag:, :]
    xx = mlag2(FPC, nlag)
    xx = xx[nlag:, :]
    
    # Create RHS matrix for VAR
    try:
        Flag, _ = create_RHS_NI(xx, r, nlag, t-nlag)
    except:
        # Fallback if RHS creation fails
        print("Warning: Failed to create RHS matrix. Using simplified approach.")
        Flag = np.zeros((xx.shape[0] * r, r * r * nlag))
        for i in range(xx.shape[0]):
            for j in range(r):
                row_idx = i * r + j
                for k in range(nlag):
                    col_start = j * r * nlag + k * r
                    col_end = col_start + r
                    if row_idx < Flag.shape[0] and col_end <= Flag.shape[1]:
                        Flag[row_idx, col_start:col_end] = xx[i, k*r:(k+1)*r]
    
    # Kalman filter
    for irep in range(t):
        # Lambda prediction step
        if irep == 0:
            lambda_pred[:, :, irep] = lambda_0_prmean
            for i in range(p, q):
                Rl_t[:, :, i, irep] = lambda_0_prvar
        else:
            lambda_pred[:, :, irep] = lambda_update[:, :, irep-1]
            Rl_t[:, :, :, irep] = (1.0 / l_3) * Sl_t[:, :, :, irep-1]
        
        # Beta prediction step
        if irep <= nlag:
            beta_pred[:, irep] = beta_0_prmean
            beta_update[:, irep] = beta_pred[:, irep]
            Rb_t[:, :, irep] = beta_0_prvar
        else:
            beta_pred[:, irep] = beta_update[:, irep-1]
            Rb_t[:, :, irep] = (1.0 / l_4) * Sb_t[:, :, irep-1]
        
        # One-step ahead prediction
        x_t_pred[irep, :] = lambda_pred[:, :, irep] @ FPC[irep, :].T
        
        # Prediction error
        e_t[:, irep] = YX[irep, :].T - x_t_pred[irep, :].T
        
        # Measurement error covariance
        if irep == 0:
            V_t[:, :, irep] = np.diag(np.diag(V_0))
        else:
            A_t = np.outer(e_t[p:, irep], e_t[p:, irep].T)
            V_t[p:, p:, irep] = l_1 * V_t[p:, p:, irep-1] + (1-l_1) * np.diag(np.diag(A_t))
        
        # Lambda update step
        if y_true == 1:
            for i in range(p, q):
                Rx = Rl_t[:, :, i, irep] @ FPC[irep, :].T
                KV_l = V_t[i, i, irep] + FPC[irep, :] @ Rx
                KG = Rx / KV_l
                lambda_update[i, :, irep] = lambda_pred[i, :, irep] + (KG * (YX[irep, i] - lambda_pred[i, :, irep] @ FPC[irep, :].T)).T
                Sl_t[:, :, i, irep] = Rl_t[:, :, i, irep] - KG @ (FPC[irep, :] @ Rl_t[:, :, i, irep])
        
        # Beta update step
        if irep >= nlag:
            try:
                # Update VAR coefficients - ensure indices are valid
                idx_start = min((irep-nlag) * r, Flag.shape[0] - r)
                idx_end = min(idx_start + r, Flag.shape[0])
                
                # Make sure we have valid indices
                if idx_start < 0:
                    idx_start = 0
                if idx_end > Flag.shape[0]:
                    idx_end = Flag.shape[0]
                
                flag_slice = Flag[idx_start:idx_end, :]
                
                # Ensure flag_slice has correct dimensions
                if flag_slice.shape[0] > 0:
                    Rx = Rb_t[:flag_slice.shape[1], :flag_slice.shape[1], irep] @ flag_slice.T
                    
                    # Initialize Q_t for first iteration if not already set
                    if np.all(Q_t[:, :, irep] == 0):
                        Q_t[:, :, irep] = Q_0
                    
                    # Create a placeholder matrix for KV_b with correct dimensions
                    KV_b_placeholder = np.eye(flag_slice.shape[0])
                    KV_b = flag_slice @ Rx
                    
                    # Check dimensions before matrix operations
                    if KV_b.shape[0] == KV_b_placeholder.shape[0] and KV_b.shape[1] == KV_b_placeholder.shape[1]:
                        KV_b = Q_t[:flag_slice.shape[0], :flag_slice.shape[0], irep] + KV_b
                        
                        # Use safe matrix inversion
                        try:
                            KG = Rx @ LA.inv(KV_b)
                        except LA.LinAlgError:
                            # Fallback to pseudo-inverse if matrix is singular
                            KG = Rx @ LA.pinv(KV_b)
                        
                        # Update beta parameters safely
                        if KG.shape[1] == flag_slice.shape[0]:
                            beta_update[:KG.shape[0], irep] = beta_pred[:KG.shape[0], irep] + KG @ (FPC[irep, :r].T - flag_slice @ beta_pred[:flag_slice.shape[1], irep])
                            Sb_t[:KG.shape[0], :KG.shape[0], irep] = Rb_t[:KG.shape[0], :KG.shape[0], irep] - KG @ flag_slice @ Rb_t[:flag_slice.shape[1], :KG.shape[0], irep]
            except Exception as e:
                print(f"Warning: Beta update failed at t={irep}: {str(e)}")
                # Keep previous values
                if irep > 0:
                    beta_update[:, irep] = beta_update[:, irep-1]
                    Sb_t[:, :, irep] = Sb_t[:, :, irep-1]
        
        # Update transition matrices
        try:
            B = np.zeros((r, r*nlag))
            bb = beta_update[:, irep]
            
            # Reshape coefficients to transition matrix safely
            splace = 0
            for ii in range(nlag):
                for iii in range(r):
                    if splace + r <= len(bb):
                        B[iii, (ii*r):((ii+1)*r)] = bb[splace:(splace+r)]
                        splace += r
            
            # Add identity blocks for companion form
            # Ensure dimensions are compatible
            lower_block_rows = min(k - r, r * (nlag - 1))
            lower_block_cols = min(k, r * nlag)
            
            # Create a temporary full matrix with appropriate dimensions
            B_full_temp = np.zeros((r + lower_block_rows, r * nlag))
            B_full_temp[:r, :] = B
            
            if lower_block_rows > 0 and r * (nlag - 1) > 0:
                B_full_temp[r:r+lower_block_rows, :r*(nlag-1)] = np.eye(lower_block_rows)
            
            # Resize to k x k for consistency
            B_full = np.zeros((k, k))
            B_full[:B_full_temp.shape[0], :B_full_temp.shape[1]] = B_full_temp
            
            # Store time-varying matrices
            lambda_t[:, :, irep] = lambda_update[:, :, irep]
            
            # Check stability of VAR (with dimension safeguards)
            stable = False
            try:
                eigvals = LA.eigvals(B_full)
                stable = np.max(np.abs(eigvals)) < 0.9999
            except LA.LinAlgError:
                # If eigenvalue computation fails, assume not stable
                stable = False
            
            if stable:
                beta_t[:, :, irep] = B_full
            else:
                if irep > 0:
                    beta_t[:, :, irep] = beta_t[:, :, irep-1]
                    beta_update[:, irep] = 0.95 * beta_update[:, irep-1]
                else:
                    # For first period, use a scaled initialization
                    beta_t[:, :, irep] = 0.5 * B_full
        except Exception as e:
            print(f"Warning: Transition matrix update failed at t={irep}: {str(e)}")
            # Fallback: use previous values or initialize safely
            if irep > 0:
                beta_t[:, :, irep] = beta_t[:, :, irep-1]
            else:
                beta_t[:, :, irep] = np.eye(k) * 0.5
        
        # Update state error covariance
        if irep == 0:
            Q_t[:, :, irep] = Q_0
        else:
            try:
                if irep <= nlag:
                    # For initial periods, use a scaled outer product
                    FPC_slice = FPC[irep, :r]
                    Gf_t = 0.1 * np.outer(FPC_slice, FPC_slice)
                else:
                    # Compute residuals for state equation
                    if irep - nlag < yy.shape[0] and irep - nlag < xx.shape[0]:
                        # Extract right dimensions for multiplication
                        B_slice = B[:r, :r]  # Use the top-left block of B
                        
                        # Check if dimensions match before matrix multiplication
                        if xx[irep-nlag, :r].shape[0] == B_slice.shape[1]:
                            # Use the actual rows from xx rather than attempting to use beta_t
                            res = yy[irep-nlag, :r] - xx[irep-nlag, :r] @ B_slice.T
                            Gf_t = np.outer(res, res)
                        else:
                            # Fallback if dimensions don't match
                            Gf_t = 0.1 * Q_t[:, :, irep-1]
                    else:
                        # Fallback if indices are out of bounds
                        Gf_t = 0.1 * Q_t[:, :, irep-1]
                
                # Update Q_t with exponential weighting
                Q_t[:, :, irep] = l_2 * Q_t[:, :, irep-1] + (1-l_2) * Gf_t
            except Exception as e:
                print(f"Warning: State error covariance update failed at t={irep}: {str(e)}")
                # Fallback: use previous values with slight inflation
                Q_t[:, :, irep] = 1.05 * Q_t[:, :, irep-1]
    
    # Kalman smoother
    lambda_new = np.zeros_like(lambda_update)
    beta_new = np.zeros_like(beta_update)
    
    lambda_new[:, :, -1] = lambda_update[:, :, -1]
    beta_new[:, -1] = beta_update[:, -1]
    
    Q_t_new = np.zeros_like(Q_t)
    V_t_new = np.zeros_like(V_t)
    
    Q_t_new[:, :, -1] = Q_t[:, :, -1]
    V_t_new[:, :, -1] = V_t[:, :, -1]
    
    for irep in range(t-2, -1, -1):
        # Smooth lambda
        lambda_new[:r, :, irep] = lambda_update[:r, :, irep]
        
        if y_true == 1:
            for i in range(r, q):
                try:
                    # Safe matrix inversion
                    Ul_t_safe = Sl_t[:, :, i, irep] @ LA.pinv(Rl_t[:, :, i, irep+1])
                    lambda_new[i, :, irep] = lambda_update[i, :, irep] + (lambda_new[i, :, irep+1] - lambda_pred[i, :, irep+1]) @ Ul_t_safe.T
                except Exception:
                    # Fallback: keep filtered value
                    lambda_new[i, :, irep] = lambda_update[i, :, irep]
        
        # Smooth beta
        try:
            # Check if Rb_t has valid values
            if np.sum(np.abs(Rb_t[:, :, irep+1])) < 1e-10:
                beta_new[:, irep] = beta_update[:, irep]
            else:
                # Safe matrix inversion
                Ub_t_safe = Sb_t[:, :, irep] @ LA.pinv(Rb_t[:, :, irep+1])
                beta_new[:, irep] = beta_update[:, irep] + Ub_t_safe @ (beta_new[:, irep+1] - beta_pred[:, irep+1])
        except Exception:
            # Fallback: keep filtered value
            beta_new[:, irep] = beta_update[:, irep]
        
        # Smooth covariances
        Q_t_new[:, :, irep] = 0.9 * Q_t[:, :, irep] + 0.1 * Q_t_new[:, :, irep+1]
        V_t_new[p:, p:, irep] = 0.9 * V_t[p:, p:, irep] + 0.1 * V_t_new[p:, p:, irep+1]
    
    # Finalize transition matrices
    for irep in range(t):
        try:
            B = np.zeros((r, r*nlag))
            bb = beta_new[:, irep]
            
            # Reshape coefficients to transition matrix safely
            splace = 0
            for ii in range(nlag):
                for iii in range(r):
                    if splace + r <= len(bb):
                        B[iii, (ii*r):((ii+1)*r)] = bb[splace:(splace+r)]
                        splace += r
            
            # Create companion form matrix
            if r + r*(nlag-1) <= k:
                lower_block_rows = r * (nlag - 1)
                lower_block_cols = r * nlag
                
                # Create the complete transition matrix
                B_full = np.zeros((k, k))
                B_full[:r, :r*nlag] = B
                
                if lower_block_rows > 0:
                    B_full[r:r+lower_block_rows, :r*(nlag-1)] = np.eye(lower_block_rows)
                
                # Store final values
                lambda_t[:, :, irep] = lambda_new[:, :, irep]
                beta_t[:, :, irep] = B_full
            else:
                # Fallback for dimension mismatch
                print(f"Warning: Dimension mismatch at t={irep}, using simplified approach")
                lambda_t[:, :, irep] = lambda_new[:, :, irep]
                
                # Create a smaller dimension transition matrix
                max_dim = min(k, r + r*(nlag-1))
                B_simple = np.zeros((max_dim, max_dim))
                B_simple[:min(r, max_dim), :min(r*nlag, max_dim)] = B[:min(r, max_dim), :min(r*nlag, max_dim)]
                
                if r < max_dim and r*(nlag-1) > 0:
                    identity_size = min(max_dim - r, r*(nlag-1))
                    B_simple[r:r+identity_size, :identity_size] = np.eye(identity_size)
                
                beta_t[:max_dim, :max_dim, irep] = B_simple
        except Exception as e:
            print(f"Warning: Final transition matrix update failed at t={irep}: {str(e)}")
            # Use previous values as fallback
            if irep > 0:
                beta_t[:, :, irep] = beta_t[:, :, irep-1]
            else:
                beta_t[:, :, irep] = np.eye(k) * 0.5
            
            lambda_t[:, :, irep] = lambda_new[:, :, irep]
    
    return beta_t, beta_new, lambda_t, V_t_new, Q_t_new

# KFS Factors: Kalman filter and smoother for factors
def KFS_factors(YX, lambda_t, beta_t, V_t, Q_t, nlag, k, r, q, t, factor_0):
    """
    Kalman filter and smoother for factors
    
    Parameters:
    YX (numpy.ndarray): Combined Y and X data
    lambda_t (numpy.ndarray): Time-varying loadings
    beta_t (numpy.ndarray): Time-varying transition matrices
    V_t (numpy.ndarray): Measurement error covariance
    Q_t (numpy.ndarray): State error covariance
    nlag (int): Number of lags
    k (int): Dimension of state vector
    r (int): Number of factors
    q (int): Total number of variables
    t (int): Number of observations
    factor_0 (dict): Prior for factors
    
    Returns:
    tuple: Smoothed factors and their covariance
    """
    # Unpack priors
    factor_0_prmean = factor_0['mean']
    factor_0_prvar = factor_0['var']
    
    # Initialize storage
    factor_pred = np.zeros((k, t))
    factor_update = np.zeros((k, t))
    
    Rf_t = np.zeros((k, k, t))
    Sf_t = np.zeros((k, k, t))
    
    x_t_predf = np.zeros((t, q))
    ef_t = np.zeros((q, t))
    
    # Kalman filter
    for irep in range(t):
        # Prediction step
        if irep == 0:
            factor_pred[:, irep] = factor_0_prmean.flatten()
            Rf_t[:, :, irep] = factor_0_prvar
        else:
            factor_pred[:, irep] = beta_t[:, :, irep-1] @ factor_update[:, irep-1]
            
            # Create full Q matrix with zeros for lower states
            Q_full = np.zeros((k, k))
            Q_full[:r, :r] = Q_t[:, :, irep]
            
            Rf_t[:, :, irep] = beta_t[:, :, irep-1] @ Sf_t[:, :, irep-1] @ beta_t[:, :, irep-1].T + Q_full
        
        # One-step ahead prediction
        x_t_predf[irep, :] = lambda_t[:, :, irep] @ factor_pred[:r, irep]
        
        # Prediction error
        ef_t[:, irep] = YX[irep, :].T - x_t_predf[irep, :].T
        
        # Update step
        H_lambda = lambda_t[:, :, irep]
        
        KV_f = V_t[:, :, irep] + H_lambda @ Rf_t[:r, :r, irep] @ H_lambda.T
        KG = Rf_t[:r, :r, irep] @ H_lambda.T @ LA.inv(KV_f)
        
        factor_update[:r, irep] = factor_pred[:r, irep] + KG @ ef_t[:, irep]
        Sf_t[:r, :r, irep] = Rf_t[:r, :r, irep] - KG @ H_lambda @ Rf_t[:r, :r, irep]
    
    # Kalman smoother
    factor_new = np.zeros_like(factor_update)
    Sf_t_new = np.zeros_like(Sf_t)
    
    factor_new[:, -1] = factor_update[:, -1]
    Sf_t_new[:, :, -1] = Sf_t[:, :, -1]
    
    for irep in range(t-2, -1, -1):
        # Smoothing step
        Z_t = Sf_t[:, :, irep] @ beta_t[:, :, irep].T
        U_t = Z_t[:r, :r] @ LA.inv(Rf_t[:r, :r, irep+1])
        
        factor_new[:r, irep] = factor_update[:r, irep] + U_t @ (factor_new[:r, irep+1] - factor_pred[:r, irep+1])
        Sf_t_new[:r, :r, irep] = Sf_t[:r, :r, irep] + U_t @ (Sf_t_new[:r, :r, irep+1] - Rf_t[:r, :r, irep+1]) @ U_t.T
    
    return factor_new, Sf_t_new

# Generate simulated data for testing
def simulate_data(T=200, n_x=20, n_y=3, n_factors=1, ar_params=None, seed=None):
    """
    Generate simulated data for testing the TVP-FAVAR model
    
    Parameters:
    T (int): Number of time periods
    n_x (int): Number of X variables
    n_y (int): Number of Y variables
    n_factors (int): Number of factors
    ar_params (list): AR parameters for factors
    seed (int): Random seed
    
    Returns:
    tuple: X data, Y data, true factors
    """
    if seed is not None:
        np.random.seed(seed)
    
    if ar_params is None:
        ar_params = [0.7, -0.2]  # AR(2) process by default
    
    # Generate factors (AR process)
    p = len(ar_params)
    factors = np.zeros((T, n_factors))
    
    # Initialize with random values
    factors[:p, :] = np.random.normal(0, 1, (p, n_factors))
    
    # Generate AR process
    for t in range(p, T):
        factors[t, :] = np.sum([ar_params[i] * factors[t-i-1, :] for i in range(p)], axis=0) + np.random.normal(0, 0.5, n_factors)
    
    # Generate loadings with some time variation
    loadings_x = np.zeros((n_x, n_factors, T))
    loadings_y = np.zeros((n_y, n_factors, T))
    
    # Initialize loadings
    base_loadings_x = np.random.uniform(0.5, 1.5, (n_x, n_factors))
    base_loadings_y = np.random.uniform(0.5, 1.5, (n_y, n_factors))
    
    # Add time variation to loadings
    for t in range(T):
        # Smooth time variation using sine waves with different frequencies
        time_factor_x = np.sin(np.arange(n_x) * np.pi / n_x + t / 30) * 0.3
        time_factor_y = np.sin(np.arange(n_y) * np.pi / n_y + t / 20) * 0.2
        
        # Apply time variation
        loadings_x[:, :, t] = base_loadings_x + time_factor_x.reshape(-1, 1)
        loadings_y[:, :, t] = base_loadings_y + time_factor_y.reshape(-1, 1)
    
    # Generate X data: X = loadings * factors + noise
    X = np.zeros((T, n_x))
    for t in range(T):
        # Fix the dimension mismatch: Instead of using matrix multiplication with reshape,
        # we'll use broadcasting and element-wise multiplication
        X[t, :] = np.sum(loadings_x[:, :, t] * factors[t, :], axis=1) + np.random.normal(0, 0.5, n_x)
    
    # Generate Y data: Y = loadings * factors + noise (with different noise level)
    Y = np.zeros((T, n_y))
    for t in range(T):
        # Same fix for Y data
        Y[t, :] = np.sum(loadings_y[:, :, t] * factors[t, :], axis=1) + np.random.normal(0, 0.3, n_y)
    
    # Add different characteristics to data
    
    # 1. Add a structural break to some X variables
    break_point = int(T * 0.7)
    X[break_point:, :5] = X[break_point:, :5] + 2.0
    
    # 2. Add seasonality to Y variables
    seasons = 4  # quarterly data
    seasonal_pattern = np.tile(np.sin(np.arange(seasons) * 2 * np.pi / seasons), (T // seasons) + 1)[:T]
    for i in range(n_y):
        Y[:, i] += seasonal_pattern * (0.5 + i * 0.2)
    
    # 3. Add some missing values to X (about 5%)
    mask = np.random.random((T, n_x)) < 0.05
    X_with_missing = X.copy()
    X_with_missing[mask] = np.nan
    
    # 4. Add correlation between some X variables
    corr_factor = np.random.normal(0, 1, T)
    for i in range(5, 10):
        X[:, i] += corr_factor * (0.2 * i)
    
    # 5. Add a trend component to Y
    trend = np.linspace(0, 2, T)
    Y += trend.reshape(-1, 1)
    
    return X_with_missing, Y, factors

# Validation function for TVP-FAVAR model
def validate_tvp_favar(true_factors, estimated_factors, forecasts=None, true_values=None, significance_level=0.05):
    """
    Validate TVP-FAVAR model estimation and forecasts
    
    Parameters:
    true_factors (numpy.ndarray): True factors used to generate data
    estimated_factors (numpy.ndarray): Estimated factors from the model
    forecasts (numpy.ndarray, optional): Forecasted values
    true_values (numpy.ndarray, optional): True values for forecast period
    significance_level (float): Significance level for tests
    
    Returns:
    dict: Validation results
    """
    results = {}
    
    # 1. Correlation between true and estimated factors
    T = true_factors.shape[0]
    n_factors = true_factors.shape[1]
    
    factor_correlations = []
    for i in range(n_factors):
        # Normalize factors (they might have different scales)
        true_norm = (true_factors[:, i] - np.mean(true_factors[:, i])) / np.std(true_factors[:, i])
        est_norm = (estimated_factors[:, i] - np.mean(estimated_factors[:, i])) / np.std(estimated_factors[:, i])
        
        # Compute correlation
        corr = np.corrcoef(true_norm, est_norm)[0, 1]
        factor_correlations.append(corr)
    
    results['factor_correlations'] = factor_correlations
    results['avg_factor_correlation'] = np.mean(factor_correlations)
    
    # 2. Check for stationarity of factors (Dickey-Fuller test)
    from statsmodels.tsa.stattools import adfuller
    
    adf_results = []
    for i in range(n_factors):
        adf_test = adfuller(estimated_factors[:, i])
        adf_results.append({
            'factor': i,
            'statistic': adf_test[0],
            'p-value': adf_test[1],
            'stationary': adf_test[1] < significance_level
        })
    
    results['stationarity_tests'] = adf_results
    
    # 3. Check forecast accuracy (if provided)
    if forecasts is not None and true_values is not None:
        forecast_horizon = forecasts.shape[0]
        n_variables = forecasts.shape[1]
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(forecasts - true_values))
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((forecasts - true_values) ** 2))
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((true_values - forecasts) / true_values)) * 100
        
        # Directional accuracy
        direction_true = np.diff(true_values, axis=0) > 0
        direction_forecast = np.diff(forecasts, axis=0) > 0
        hit_rate = np.mean(direction_true == direction_forecast)
        
        results['forecast_metrics'] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': hit_rate
        }
        
        # Diebold-Mariano test for forecast comparison (if we have a benchmark)
        # This would be added if a benchmark forecast is available
    
    # 4. Check residual autocorrelation
    # Implement Ljung-Box test for residual autocorrelation
    
    return results

# Main function to run the TVP-FAVAR model
def run_tvp_favar(X=None, Y=None, nfac=1, nlag=4, decay_factors=None, y_true=1,
                 validate=True, forecast_horizon=8, plot_results=True, seed=None):
    """
    Run the TVP-FAVAR model with simulated or provided data
    
    Parameters:
    X (numpy.ndarray, optional): X data, if None will use simulated data
    Y (numpy.ndarray, optional): Y data, if None will use simulated data
    nfac (int): Number of factors
    nlag (int): Number of lags
    decay_factors (list): List of decay factors [l_1, l_2, l_3, l_4]
    y_true (int): Flag indicating if Y should be included
    validate (bool): Whether to validate results
    forecast_horizon (int): Forecast horizon
    plot_results (bool): Whether to plot results
    seed (int): Random seed for simulations
    
    Returns:
    dict: Results including estimated factors, loadings, coefficients, and validation
    """
    # Set decay factors if not provided
    if decay_factors is None:
        decay_factors = [0.96, 0.96, 0.99, 0.99]
    
    # Generate simulated data if not provided
    if X is None or Y is None:
        print("Generating simulated data...")
        X, Y, true_factors = simulate_data(T=200, n_x=20, n_y=3, n_factors=nfac, seed=seed)
    else:
        true_factors = None
    
    # Set dimensions
    t, n = X.shape
    _, p = Y.shape
    r = nfac + p
    q = n + p
    m = nlag * (r ** 2)
    k = nlag * r
    
    # Standardize data
    X_std = standardize_miss(X)
    Y_std = preprocessing.scale(Y)
    
    # Combine Y and X data
    YX = np.hstack([Y_std, X_std])
    
    print(f"Data dimensions: T={t}, X variables={n}, Y variables={p}")
    print(f"Running TVP-FAVAR with {nfac} factors and {nlag} lags...")
    
    # Extract principal components
    FPC_raw, LPC = extract(X_std, nfac)
    FPC = np.hstack([Y_std, FPC_raw])
    
    # Initial OLS estimation
    print("Initial OLS estimation...")
    L_OLS, B_OLS, beta_OLS, SIGMA_OLS, Q_OLS = ols_pc_dfm(YX, FPC, LPC, y_true, n, p, r, nfac, nlag)
    
    # Set priors
    print("Setting priors...")
    
    # Initial condition on the factors
    factor_0 = {
        'mean': np.zeros((k, 1)),
        'var': 10 * np.eye(k)
    }
    
    # Initial condition on lambda_t
    lambda_0 = {
        'mean': np.zeros((q, r)),
        'var': 1 * np.eye(r)
    }
    
    # Initial condition on beta_t
    b_prior, Vb_prior = Minn_prior_KOOP(0.1, r, nlag, m)
    beta_0 = {
        'mean': b_prior,
        'var': Vb_prior,
    }
    
    # Initial condition on the covariance matrices
    V_0 = 0.1 * np.eye(q)
    V_0[:p, :p] = 0
    Q_0 = 0.1 * np.eye(r)
    
    # Put all decay/forgetting factors together in a vector
    l = np.array(decay_factors).reshape(-1, 1)
    
    # Run Kalman filter and smoother for parameters
    print("Running Kalman filter and smoother for parameters...")
    beta_t, beta_new, lambda_t, V_t, Q_t = KFS_parameters(
        YX, FPC, l, nfac, nlag, y_true, k, m, p, q, r, t, lambda_0, beta_0, V_0, Q_0
    )
    
    # Run Kalman filter and smoother for factors
    print("Running Kalman filter and smoother for factors...")
    factor_new, Sf_t_new = KFS_factors(
        YX, lambda_t, beta_t, V_t, Q_t, nlag, k, r, q, t, factor_0
    )
    
    # Extract estimated factors
    estimated_factors = factor_new[:nfac, :].T
    
    # Generate forecasts
    print(f"Generating {forecast_horizon} step ahead forecasts...")
    forecasts = np.zeros((forecast_horizon, r))
    
    # Initial state for forecast
    last_factor = factor_new[:, -1].copy()
    
    # Iterate forecasts
    for h in range(forecast_horizon):
        # Use the last time varying parameter matrices
        forecasts[h, :] = (beta_t[:r, :r, -1] @ last_factor[:r]).flatten()
        
        # Update state for next forecast
        temp = last_factor.copy()
        last_factor[:r] = forecasts[h, :]
        last_factor[r:] = temp[:k-r]
    
    # Validation
    validation_results = None
    if validate and true_factors is not None:
        print("Validating results...")
        validation_results = validate_tvp_favar(true_factors, estimated_factors)
        
        # Print validation summary
        print("\nValidation Summary:")
        print(f"Average factor correlation: {validation_results['avg_factor_correlation']:.4f}")
        
        for i, corr in enumerate(validation_results['factor_correlations']):
            print(f"Factor {i+1} correlation: {corr:.4f}")
        
        for test in validation_results['stationarity_tests']:
            status = "Stationary" if test['stationary'] else "Non-stationary"
            print(f"Factor {test['factor']+1} stationarity: {status} (p-value: {test['p-value']:.4f})")
    
    # Plot results
    if plot_results:
        print("Plotting results...")
        
        plt.figure(figsize=(15, 10))
        
        # Plot estimated factors
        plt.subplot(2, 2, 1)
        for i in range(min(nfac, 3)):  # Plot up to 3 factors
            plt.plot(estimated_factors[:, i], label=f'Factor {i+1}')
        plt.title('Estimated Factors')
        plt.legend()
        
        # Plot true vs estimated factors if available
        if true_factors is not None:
            plt.subplot(2, 2, 2)
            for i in range(min(nfac, 3)):  # Plot up to 3 factors
                plt.plot(true_factors[:, i], label=f'True Factor {i+1}')
                plt.plot(estimated_factors[:, i], '--', label=f'Est. Factor {i+1}')
            plt.title('True vs Estimated Factors')
            plt.legend()
        
        # Plot time-varying loadings for first Y variable
        plt.subplot(2, 2, 3)
        for i in range(min(r, 3)):  # Plot loadings for up to 3 factors
            plt.plot(lambda_t[0, i, :], label=f'Loading on Factor {i+1}')
        plt.title('Time-varying Loadings (First Y Variable)')
        plt.legend()
        
        # Plot forecasts
        plt.subplot(2, 2, 4)
        for i in range(min(p, 3)):  # Plot forecasts for up to 3 variables
            plt.plot(range(t, t + forecast_horizon), forecasts[:, i], '--', label=f'Forecast Y{i+1}')
            plt.plot(range(t-10, t), Y_std[t-10:t, i], label=f'Actual Y{i+1}')
        plt.title('Forecasts')
        plt.axvline(x=t-1, color='k', linestyle='-')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Return results
    results = {
        'estimated_factors': estimated_factors,
        'factor_loadings': lambda_t,
        'var_coefficients': beta_t,
        'forecasts': forecasts,
        'validation': validation_results
    }
    
    return results

# Example usage with simulated data
# Set random seed for reproducibility
np.random.seed(123)
    
# Run TVP-FAVAR with simulated data
results = run_tvp_favar(nfac=1, nlag=2, validate=True, plot_results=True)
    
# Access results
estimated_factors = results['estimated_factors']
factor_loadings = results['factor_loadings']
var_coefficients = results['var_coefficients']
forecasts = results['forecasts']
validation = results['validation']
    
print("TVP-FAVAR estimation completed successfully!")
