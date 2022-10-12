# Python Functions to forecast inflation using TVP-FAVAR
# Kebutuhan functions
#1. standardize_miss (done)
#2. standardize (done)
#3. Minn_prior_KOOP (done)
#4. extract (done)
#5. olssvd (done)
#6. mlag2 (done)
#7. ols_pc_dfm (done)
#8. Kalman_companion (done)
#9. create_RHS_ini (done)
#10. KFS_parameters (done)
#11. KFS_factors (done)    

# Import library
import numpy as np
import pandas as pd
import numpy.matlib
from numpy import linalg as LA
from sklearn import preprocessing
    
# Minn Prior Koop
def Minn_prior_KOOP(gamma = None,M = None,p = None,K = None):
    # gamma : float, KoopKorobilis(2014) = 0.1
    # M : int, KoopKorobilis(2014) = 4
    # p : int, KoopKorobilis(2014) = 4 (number of lag)
    # K : int, KoopKorobilis(2014) = 4
    
    # This is the version of the Minnesota prior with no dependence on the
    # standard deviations of the univariate regressions. This prior allows
    # online estimation and forecasting of the large TVP-VAR.
    
    # 1. Minnesota Mean on VAR regression coefficients
    A_prior = np.transpose(np.array([[0.9 * np.eye(M)],[np.zeros(((p - 1) * M,M))]]))
    a_prior = (A_prior)
    # 2. Minnesota Variance on VAR regression coefficients
    
    # Create an array of dimensions K x M, which will contain the K diagonal
    # elements of the covariance matrix, in each of the M equations.
    V_i = np.zeros((int(K / M),M))
    
    for i in np.arange(1,M+1).reshape(-1):
        for j in np.arange(1,K / M+1).reshape(-1):
            i_np = i - 1
            j_np = j - 1
            V_i[int(j_np),i_np] = gamma / ((np.ceil(j / M)) ** 2)
            # Note: the "ceil((j-1)/M^2)" command finds the associated lag
            # number for each parameter
    
    # Now V (MINNESOTA VARIANCE) is a diagonal matrix with diagonal elements the transpose(V_i)
    V_i_T = np.transpose(V_i)
    V_prior = np.diag(V_i_T) # dicek lagi
    
    return a_prior,V_prior


# Extract (jalan, hasil cukup dekat)
def extract(data = None,k = None): 
    # data : matrix, KoopKorobilis(2014) = matrix 213x20 (matrix yang berisi data)
    # k : int, KoopKorobilis(2014) = 1
    
    t,n = data.shape
    xx = np.matmul(np.transpose(data), data)
    w, v = LA.eig(xx)
    evec = v
    eval_val = w * np.identity(len(w))
    
    # evec harusnya diagonal matrix D of eigenvalues
    # eval_val harusnya matrix V whose columns are the corresponding right eigenvectors
    
    # sorting evec so that they correspond to eval in descending order
    eval_val = np.sort(np.diag(eval_val)) # command untuk short
    index = np.linspace(1, len(np.sort(np.diag(eval_val))), num=len(np.sort(np.diag(eval_val))))
    indeks = np.flipud(index) # flip array up to down
    
    evc = np.zeros((n,n))
    for i in range(n):
        evc[:,i] = evec[:,int(indeks[i])-1]
    
    lam = np.sqrt(n) * evc[:,0:k] # ini dicek
    fac = np.matmul(data ,(lam / n)) # ini dicek
    return fac,lam

# olssvd (jalan, hasil cukup dekat)
def olssvd(y = None,ly = None):
    # y : matrix, KoopKorobilis(2014) = matrix 213x23
    # ly : matrix, KoopKorobilis(2014) = matrix 213x4
    
    vl,d,vr = LA.svd(ly, full_matrices=False)
    d_mat = d * np.identity(len(d))
    
    d_mat2 = 1.0 / np.diag(d_mat)
    b = np.matmul((np.multiply(vr,np.matlib.repmat(np.transpose(d_mat2),vr.shape[1-1],1))), (np.matmul(np.transpose(vl), y)))
    return b

# mlag2 (jalan, ada beberapa row dan column yg salah (jauh hasilnya))
def mlag2(X = None,p = None): 
    # X : matrix, KoopKorobilis(2014) = matrix 213x4
    # p : int, KoopKorobilis(2014) = 4
    
    # MLAG2 Summary of this function goes here
    # Detailed explanation goes here
    Traw,N = X.shape
    Xlag = np.zeros((Traw,N * p))
    
    for ii in range(1, p+1):
        Xlag[p:Traw, (N*(ii-1)):(N*ii)] = X[(p-ii):(Traw-ii),:N]
    
    # #OR:
    # [Traw,N]=size(X);
    # Xlag=zeros(Traw,N,p);
    # for ii=1:p
    #     Xlag(p+1:Traw,:,ii)=X(p+1-ii:Traw-ii,:);
    # end
    # Xlag=Xlag(:,:);
    return Xlag

# OLS PC DFM (jalan, hasilnya agak jauh, sumber kesalahan ada di function mlag2)
def ols_pc_dfm(YX = None,YF = None,Lf = None,y_true = None,n = None,p = None,r = None,
               nfac = None, nlag = None): 
    # YX: matrix, KoopKorobilis(2014) = matrix 213x23
    # YF : matrix, KoopKorobilis(2014) = matrix 213x4
    # Lf : matrix, KoopKorobilis(2014) = matrix 20x1
    # y_true : int, KoopKorobilis(2014) = 1
    # n : int, KoopKorobilis(2014) = 20 (the number of variables)
    # p : int, KoopKorobilis(2014) = 3
    # r : int, KoopKorobilis(2014) = 4
    # nfac : int, KoopKorobilis(2014) = 1
    # nlag : int, KoopKorobilis(2014) = 4 (the number of lags)
    # ols_pc_dfm(YX,FPC,LPC,y_true,n,p,r,nfac,nlag)
    
    # YX = YX
    # YF = FPC
    # Lf = LPC
    # y_true = y_true
    # n = n
    # p = p
    # r = r
    # nfac = nfac
    # nlag = nlag
    
    
    #Lf = Lf./Lf(1,1);
    t = YX.shape[1-1]
    # Obtain L (the loadings matrix)
    if y_true == 1:
        L = np.transpose(olssvd(YX,YF))
    elif y_true == 0:
        L = np.array([[np.eye(p),np.zeros((p,nfac))],[np.zeros((n,p)),Lf]])
    else:
        print('unrecognized y_true')
    
    # Obtain the errors of the factor equation
    e = YX - np.matmul(YF, np.transpose(L))
    sigma2 = np.diag(np.diag(np.matmul(np.transpose(e), e) / t))
    # Obtain the errors of the VAR(1) equation
    yy = YF[nlag:,:]
    xx = mlag2(YF,nlag)
    xx = xx[nlag:,:]
    beta_OLS = (LA.inv(np.matmul(np.transpose(xx), xx))) @ (np.transpose(xx) @ yy)
    # beta_OLS = np.matmul(LA.inv(np.matmul(np.transpose(xx), xx)), np.matmul(np.transpose(xx), yy))
    sigmaf = np.transpose(yy - xx @ beta_OLS) @ (yy - xx @ beta_OLS) / (t - nlag - 1)
    # beta_var = np.kron(sigmaf,LA.inv(np.matmul(np.transpose(xx), xx)))
    bb = []
    for i in range(1, nlag+1):
        g = beta_OLS[(((i-1)*r+1)-1):i*r, :r]
        g_flat = g.flatten()
        bb = np.concatenate([bb,g_flat], axis = 0)
    
    return L,bb,beta_OLS,sigma2,sigmaf

# Kalman Companion (belum ditest)
def Kalman_companion(data = None,S0 = None,P0 = None,H = None,R = None,F = None,Q = None):
    # data: matrix, KoopKorobilis(2014) = matrix 213x23
    # S0: matrix, KoopKorobilis(2014) = matrix 16x1
    # P0: matrix, KoopKorobilis(2014) = matrix 16x16
    # H: matrix, KoopKorobilis(2014) = matrix 23x4
    # R: matrix, KoopKorobilis(2014) = matrix 23x23
    # F: matrix, KoopKorobilis(2014) = matrix 16x16
    # Q: matrix, KoopKorobilis(2014) = matrix 16x16
    
    
    t,nm = data.shape
    kml = S0.shape[1-1]
    km = H.shape[2-1]
    # KF
    Sp = S0
    
    Pp = P0
    S = np.zeros((t,kml))
    P = np.zeros((kml ** 2,t))
    for i in np.arange(1,t+1).reshape(-1):
        y = np.transpose(data[i,:])
        nu = y - H * Sp(np.arange(1,km+1))
        f = H * Pp(np.arange(1,km+1),np.arange(1,km+1)) * np.transpose(H) + R
        finv = np.transpose(H) / f
        Stt = Sp + np.matmul(np.matmul(Pp[:,1:km+1], finv), nu)
        Ptt = Pp - np.matmul(np.matmul(Pp[:,1:km+1], finv), (H * Pp[1:km+1,:]))
        if i < t:
            Sp = F * Stt
            Pp = F * Ptt * np.transpose(F) + Q
        S[i,:] = np.transpose(Stt)
        P[:,i] = Ptt.reshape((kml ** 2,1)) # reshape(Ptt,kml ** 2,1) # reshape jadi matrix 256x1
    
    # draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
    Sdraw = np.zeros((t,kml))
    Sdraw[t,:] = S[t,:]
    # iterate 'down', drawing at each step, use modification for singular Q
    Qstar = Q(np.arange(1,km+1),np.arange(1,km+1))
    Fstar = F[1:km+1,:]
    for i in np.arange(1,t - 1+1).reshape(-1):
        Sf = np.transpose(Sdraw(t - i + 1,np.arange(1,km+1)))
        Stt = np.transpose(S[t - i,:])
        Ptt = P[:,t - i].reshpae((kml,kml))  # reshape(P[:,t - i],kml,kml)
        f = Fstar * Ptt * np.transpose(Fstar) + Qstar
        finv = np.transpose(Fstar) / f
        nu = Sf - Fstar * Stt
        Smean = Stt + Ptt * finv * nu
        # Svar = Ptt - Ptt * finv * (Fstar * Ptt)
        Sdraw[t - i,:] = np.transpose(Smean)
    
    Sdraw = Sdraw[:,1:km+1]
    return Sdraw

# create_RHS_ini (belum ditest)
def create_RHS_NI(YY = None,M = None,p = None,t = None):
    # YY : matrix, KoopKorobilis(2014) = matrix 209x16
    # M : int, KoopKorobilis(2014) = int = 4
    # p : int, KoopKorobilis(2014) = int = 4
    # t : int, KoopKorobilis(2014) = int = 213
    
    # No intercept case
    
    K = p * (M ** 2)
    
    # Create x_t matrix.
    # first find the zeros in matrix x_t
    x_t = np.zeros(((t - p) * M,K))
    for i in np.arange(1,t - p+1).reshape(-1):
        ztemp = []
        for j in np.arange(1,p+1).reshape(-1):
            xtemp = YY(i,np.arange((j - 1) * M + 1,j * M+1))
            xtemp = np.kron(np.eye(M),xtemp)
            ztemp = np.array([ztemp,xtemp])
        x_t[np.arange[[i - 1] * M + 1,i * M+1],:] = ztemp
    
    return x_t,K

# KFS_parameters
    
def KFS_parameters(YX = None,FPC = None,l = None,nfac = None,nlag = None,y_true = None,k = None,m = None,
                   p = None,q = None,r = None,t = None,lambda_0 = None,beta_0 = None,V_0 = None,Q_0 = None): 
    # YX : matrix, KoopKorobilis(2014) = matrix 213x23
    # FPC : matrix, KoopKorobilis(2014) = matrix 213x4
    # l : matrix, KoopKorobilis(2014) = matrix 4x1
    # nfac : int, KoopKorobilis(2014) = 1
    # nlag : int, KoopKorobilis(2014) = 4
    # y_true : int, KoopKorobilis(2014) = 1
    # k : int, KoopKorobilis(2014) = 16
    # m : int, KoopKorobilis(2014) = 64
    # p : int, KoopKorobilis(2014) = 3
    # q : int, KoopKorobilis(2014) = 23
    # r : int, KoopKorobilis(2014) = 4
    # t : int, KoopKorobilis(2014) = 213
    # lambda_0 : list, KoopKorobilis(2014) berisi mean (matrix 23 x 4) var (matrix 4 x 4)
    # beta_0 : list, KoopKorobilis(2014) berisi mean (matrix 64 x 1) var (matrix 64 x 64)
    # V_0 : matrix, KoopKorobilis(2014) = matrix 23x23
    # Q_0 : matrix, KoopKorobilis(2014) = matrix 4x4
    
    
    # Function to estimate time-varying loadings, coefficients, and covariances
    # from a TVP-FAVAR, conditional on feeding in an estimate of the factors
    # (Principal Components). This function runs the Kalman filter and smoother
    # for all time-varying parameters using an adaptive algorithm (EWMA filter
    # for the covariances).
    
    # Written by Fawdy, Adapted from Dimitris Korobili (2012)
    # University of Glasgow
    
    # Initialize matrices
    lambda_0_prmean = lambda_0.mean
    lambda_0_prvar = lambda_0.var
    beta_0_prmean = beta_0.mean
    beta_0_prvar = beta_0.var
    lambda_pred = np.zeros((q,r,t))
    lambda_update = np.zeros((q,r,t))
    for j in np.arange(1,t+1).reshape(-1):
        lambda_pred[np.arange[1,r+1],np.arange[1,r+1],j] = np.eye(r)
        lambda_update[np.arange[1,r+1],np.arange[1,r+1],j] = np.eye(r)
    
    beta_pred = np.zeros((m,t))
    beta_update = np.zeros((m,t))
    Rl_t = np.zeros(r,r,q,t)
    Sl_t = np.zeros(r,r,q,t)
    Rb_t = np.zeros((m,m,t))
    Sb_t = np.zeros((m,m,t))
    x_t_pred = np.zeros((t,q))
    e_t = np.zeros((q,t))
    lambda_t = np.zeros((q,r,t))
    beta_t = np.zeros((k,k,t))
    Q_t = np.zeros((r,r,t))
    V_t = np.zeros((q,q,t))
    # Decay and forgetting factors
    l_1 = l(1)
    l_2 = l(2)
    l_3 = l(3)
    l_4 = l(4)
    # Define lags of the factors to be used in the state (VAR) equation
    yy = FPC[np.arange(nlag + 1,t+1),:]
    xx = mlag2(FPC,nlag)
    xx = xx[np.arange(nlag + 1,t+1),:]
    templag = mlag2(FPC,nlag)
    templag = templag[np.arange(nlag + 1,t+1),:]
    Flagtemp,m = create_RHS_NI(templag,r,nlag,t)
    Flag = np.array([[np.zeros((k,m))],[Flagtemp]])
    
    
    # ======================| 1. KALMAN FILTER
    for irep in np.arange(1,t+1).reshape(-1):
        
        # Assign coefficients
        bb = beta_update[:,irep]
        splace = 0
        biga = 0
        for ii in np.arange(1,nlag+1).reshape(-1):
            for iii in np.arange(1,r+1).reshape(-1):
                biga[iii,np.arange[[ii - 1] * r + 1,ii * r+1]] = np.transpose(bb(np.arange(splace + 1,splace + r+1),1))
                splace = splace + r
        B = np.array([[biga],[np.eye(r * (nlag - 1)),np.zeros((r * (nlag - 1),r))]])
        #B = [reshape(beta_update(:,irep),r,r*nlag) ; eye(r*(nlag-1)) zeros(r*(nlag-1),r)];
        lambda_t[:,:,irep] = lambda_update[:,:,irep]
        if np.amax(np.abs(LA.eig(B))) < 0.9999:
            beta_t[:,:,irep] = B
        else:
            beta_t[:,:,irep] = beta_t[:,:,irep - 1]
            beta_update[:,irep] = 0.95 * beta_update[:,irep - 1]
        
        # -----| Update the state covariances
        # 1. Get the variance of the factor
        # Update Q[t]
        if irep == 1:
            Q_t[:,:,irep] = Q_0
        elif irep > 1:
            if irep <= (nlag + 1):
                Gf_t = 0.1 * np.matmul(np.transpose(FPC[irep,:]), FPC[irep,:])
            else:
                Gf_t = np.transpose((yy[(irep - nlag),:] - xx[(irep - nlag),:] * np.transpose(B[np.arange(1,r+1),np.arange(1,k+1)]))) \
                    * (yy[(irep - nlag),:] - xx[(irep - nlag),:] \
                       * np.transpose(B[np.arange(1,r+1),np.arange(1,k+1)]))
            Q_t[:,:,irep] = l_2 * Q_t[:,:,irep - 1] + (1 - l_2) * Gf_t[np.arange(1,r+1),np.arange(1,r+1)]
        
        # =======| Kalman predict steps
        #  -for lambda
        if irep == 1:
            lambda_pred[:,:,irep] = lambda_0_prmean
            for i in np.arange(p + 1,q+1).reshape(-1):
                Rl_t[:,:,i,irep] = lambda_0_prvar
        else:
            if irep > 1:
                lambda_pred[:,:,irep] = lambda_update[:,:,irep - 1]
                Rl_t[:,:,:,irep] = (1.0 / l_3) * Sl_t[:,:,:,irep - 1]
        # -for beta
        if irep <= nlag + 1:
            beta_pred[:,irep] = beta_0_prmean
            beta_update[:,irep] = beta_pred[:,irep]
            Rb_t[:,:,irep] = beta_0_prvar
        else:
            if irep > nlag + 1:
                beta_pred[:,irep] = beta_update[:,irep - 1]
                Rb_t[:,:,irep] = (1.0 / l_4) * Sb_t[:,:,irep - 1]
        # One step ahead prediction based on PC factor
        x_t_pred[irep,:] = lambda_pred[:,:,irep] * np.transpose(FPC[irep,:])
        # Prediction error
        e_t[:,irep] = np.transpose(YX[irep,:]) - np.transpose(x_t_pred[irep,:])
        # 3. Get the measurement error variance
        A_t = e_t[(p+1):,irep] * np.transpose(e_t[(p+1):,irep])
        if irep == 1:
            V_t[:,:,irep] = np.diag(np.diag(V_0))
        else:
            V_t[(p+1):, (p+1):, irep] = l_1 * V_t[(p+1):,(p+1):,irep-1] + (1-l_1) * np.diag(np.diag(A_t))
        
        # =======| Kalman update steps
        # -for lambda
        if y_true == 0:
            # 1/ Update loadings conditional on Principal Components estimates
            for i in np.arange(p + 1,q+1).reshape(-1):
                Rx = Rl_t(r,r,i,irep) * np.transpose(FPC(irep,r))
                KV_l = V_t(i,i,irep) + FPC(irep,r) * Rx
                KG = Rx / KV_l
                lambda_update[i,r,irep] = lambda_pred(i,r,irep) + np.transpose((KG * (np.transpose(YX(irep,i)) - lambda_pred(i,r,irep) * np.transpose(FPC(irep,r)))))
                Sl_t[r,r,i,irep] = Rl_t(r,r,i,irep) - KG * (FPC(irep,r) * Rl_t(r,r,i,irep))
        else:
            if y_true == 1:
                # 1/ Update loadings conditional on Principal Components estimates
                for i in np.arange(p + 1,q+1).reshape(-1):
                    Rx = Rl_t(np.arange(1,r+1),np.arange(1,r+1),i,irep) * np.transpose(FPC(irep,np.arange(1,r+1)))
                    KV_l = V_t(i,i,irep) + FPC(irep,np.arange(1,r+1)) * Rx
                    KG = Rx / KV_l
                    lambda_update[i,np.arange[1,r+1],irep] = lambda_pred(i,np.arange(1,r+1),irep) + np.transpose((KG * (np.transpose(YX(irep,i)) - lambda_pred(i,np.arange(1,r+1),irep) * np.transpose(FPC(irep,np.arange(1,r+1))))))
                    Sl_t[np.arange[1,r+1],np.arange[1,r+1],i,irep] = Rl_t(np.arange(1,r+1),np.arange(1,r+1),i,irep) - KG * (FPC(irep,np.arange(1,r+1)) * Rl_t(np.arange(1,r+1),np.arange(1,r+1),i,irep))
        
        # -for beta
        if irep >= nlag + 1:
            # 2/ Update VAR coefficients conditional on Principal Componets estimates
            Rx = Rb_t[:,:,irep] * np.transpose(Flag[np.arange((irep - 1) * r + 1,irep * r+1),:])
            KV_b = Q_t[:,:,irep] + Flag[np.arange((irep - 1) * r + 1,irep * r+1),:] * Rx
            KG = Rx / KV_b
            beta_update[:,irep] = beta_pred[:,irep] + (KG * (np.transpose(FPC[irep,:]) - Flag[np.arange((irep - 1) * r + 1,irep * r+1),:] * beta_pred[:,irep]))
            Sb_t[:,:,irep] = Rb_t[:,:,irep] - KG * (Flag[np.arange((irep - 1) * r + 1,irep * r+1),:] * Rb_t[:,:,irep])
        
    
    # ======================| 2. KALMAN SMOOTHER
    lambda_new = 0 * lambda_update
    beta_new = 0 * beta_update
    lambda_new[:,:,t] = lambda_update[:,:,t]
    beta_new[:,t] = beta_update[:,t]
    Q_t_new = 0 * Q_t
    Q_t_new[:,:,t] = Q_t[:,:,t]
    V_t_new = 0 * V_t
    V_t_new[:,:,t] = V_t[:,:,t]
    for irep in np.arange(t - 1,1+- 1,- 1).reshape(-1):
        # 1\ Smooth lambda
        lambda_new[np.arange[1,r+1],:,irep] = lambda_update[np.arange(1,r+1),:,irep]
        if y_true == 1:
            for i in np.arange(r + 1,q+1).reshape(-1):
                Ul_t = Sl_t(np.arange(1,r+1),np.arange(1,r+1),i,irep) / Rl_t(np.arange(1,r+1),np.arange(1,r+1),i,irep + 1)
                lambda_new[i,np.arange[1,r+1],irep] = lambda_update(i,np.arange(1,r+1),irep) + (lambda_new(i,np.arange(1,r+1),irep + 1) - lambda_pred(i,np.arange(1,r+1),irep + 1)) * np.transpose(Ul_t)
        else:
            if y_true == 0:
                for i in np.arange(r + 1,q+1).reshape(-1):
                    Ul_t = Sl_t(r,r,i,irep) / Rl_t(r,r,i,irep + 1)
                    lambda_new[i,r,irep] = lambda_update(i,r,irep) + (lambda_new(i,r,irep + 1) - lambda_pred(i,r,irep + 1)) * np.transpose(Ul_t)
        # 2\ Smooth beta
        if sum(sum(Rb_t[:,:,irep + 1])) == 0:
            beta_new[:,irep] = beta_update[:,irep]
        else:
            Ub_t = Sb_t[:,:,irep] / Rb_t[:,:,irep + 1]
            beta_new[:,irep] = beta_update[:,irep] + Ub_t * (beta_new[:,irep + 1] - beta_pred[:,irep + 1])
        # 3\ Smooth Q_t
        Q_t_new[:,:,irep] = 0.9 * Q_t[:,:,irep] + 0.1 * Q_t_new[:,:,irep + 1]
        # 4\ Smooth V_t
        V_t_new[np.arange[(p + 1):],np.arange[(p + 1):],irep] = 0.9 * V_t[(p+1):,(p+1):,irep] + 0.1 * V_t_new[(p+1):,(p+1):,(irep+1)]
    
    # Assign coefficients
    for irep in np.arange(1,t+1).reshape(-1):
        bb = beta_new[:,irep]
        splace = 0
        biga = 0
        for ii in np.arange(1,nlag+1).reshape(-1):
            for iii in np.arange(1,r+1).reshape(-1):
                biga[iii,np.arange[[ii - 1] * r + 1,ii * r+1]] = np.transpose(bb(np.arange(splace + 1,splace + r+1),1))
                splace = splace + r
        B = np.array([[biga],[np.eye(r * (nlag - 1)),np.zeros((r * (nlag - 1),r))]])
        lambda_t[:,:,irep] = lambda_new[:,:,irep]
        beta_t[:,:,irep] = B
    
    return beta_t,beta_new,lambda_t,V_t,Q_t


# KFS_factors
def KFS_factors(YX = None, lambda_t = None, beta_t = None, V_t = None, Q_t = None, nlag = None, k = None, r = None,
                q = None, t = None, factor_0 = None): 
    # YX : matrix, KoopKorobilis(2014) = matrix 213x23
    # lambda_t : 3d array, KoopKorobilis(2014) = matrix 23x4x213
    # beta_t : 3d array. KoopKorobilis(2014) = matrix 16x16x23
    # V_t : 3d array. KoopKorobilis(2014) = matrix 23x23x213
    # Q_t : 3d array. KoopKorobilis(2014) = matrix 4x4x213
    # nlag : int, KoopKorobilis(2014) = 4
    # k : int, KoopKorobilis(2014) = 16
    # r : int, KoopKorobilis(2014) = 4
    # q : int, KoopKorobilis(2014) = 23
    # t : int, KoopKorobilis(2014) = 213
    # lambda_0 : list, KoopKorobilis(2014) berisi mean (matrix 16 x 1) var (matrix 16 x 16)
    
    # Initialize matrices
    factor_0_prmean = factor_0.mean
    factor_0_prvar = factor_0.var
    factor_pred = np.zeros((k,t))
    factor_update = np.zeros((k,t))
    Rf_t = np.zeros((k,k,t))
    Sf_t = np.zeros((k,k,t))
    x_t_predf = np.zeros((t,q))
    ef_t = np.zeros((q,t))
    
    # ======================| 1. KALMAN FILTER
    for irep in np.arange(1,t+1).reshape(-1):
        # ==============|Update factors conditional on (tvp) coefficients|======
        # =======| Kalman predict step for f
        if irep == 1:
            factor_pred[:,irep] = factor_0_prmean
            Rf_t[:,:,irep] = factor_0_prvar
        else:
            if irep > 1:
                factor_pred[:,irep] = beta_t[:,:,irep - 1] * factor_update[:,irep - 1]
                Rf_t[:,:,irep] = beta_t[:,:,irep - 1] * Sf_t[:,:,irep - 1] * np.transpose(beta_t[:,:,irep - 1]) + np.array([[Q_t[:,:,irep],np.zeros((r,r * (nlag - 1)))],[np.zeros((r * (nlag - 1),r * nlag))]])
        # One step ahead prediction based on Kalman factor
        x_t_predf[irep,:] = lambda_t[:,:,irep] * factor_pred(np.arange(1,r+1),irep)
        # Prediction error
        ef_t[:,irep] = np.transpose(YX[irep,:]) - np.transpose(x_t_predf[irep,:])
        # =======| Kalman update step for f
        # 3/ Update the factors conditional on the estimate of lambda_t and beta_t
        KV_f = V_t[:,:,irep] + lambda_t[:,:,irep] * Rf_t(np.arange(1,r+1),np.arange(1,r+1),irep) * np.transpose(lambda_t[:,:,irep])
        KG = (Rf_t(np.arange(1,r+1),np.arange(1,r+1),irep) * np.transpose(lambda_t[:,:,irep])) / KV_f
        factor_update[np.arange[1,r+1],irep] = factor_pred(np.arange(1,r+1),irep) + KG * ef_t[:,irep]
        Sf_t[np.arange[1,r+1],np.arange[1,r+1],irep] = Rf_t(np.arange(1,r+1),np.arange(1,r+1),irep) - KG * (lambda_t[:,:,irep] * Rf_t(np.arange(1,r+1),np.arange(1,r+1),irep))
    
    # ======================| 2. KALMAN SMOOTHER
    # RauchTungStriebel fixed-interval smoother for the factors
    factor_new = 0 * factor_update
    Sf_t_new = 0 * Sf_t
    factor_new[:,t] = factor_update[:,t]
    Sf_t_new[:,:,t] = Sf_t[:,:,t]
    for irep in np.arange(t - 1,1+- 1,- 1).reshape(-1):
        Z_t = (Sf_t[:,:,irep] * np.transpose(beta_t[:,:,irep]))
        U_t = np.squeeze(Z_t(np.arange(1,r+1),np.arange(1,r+1)) / Rf_t(np.arange(1,r+1),np.arange(1,r+1),irep + 1))
        factor_new[np.arange[1,r+1],irep] = factor_update(np.arange(1,r+1),irep) + U_t * (factor_new(np.arange(1,r+1),irep + 1) - factor_pred(np.arange(1,r+1),irep + 1))
        Sf_t_new[np.arange[1,r+1],np.arange[1,r+1],irep] = Sf_t(np.arange(1,r+1),np.arange(1,r+1),irep) + U_t * (Sf_t(np.arange(1,r+1),np.arange(1,r+1),irep + 1) - Rf_t(np.arange(1,r+1),np.arange(1,r+1),irep + 1)) * np.transpose(U_t)
    
    return factor_new,Sf_t_new


# Running TVP FAVAR
# Function yang diperlukan dalam script running ini ada di script TVP FAVAR Function

# TVP_FAVAR - Time-varying parameters factor-augmented VAR using EWMA Kalman filters 
# SINGLE MODEL CASE
#-----------------------------------------------------------------------------------------
# The model is:
#     _    _     _              _     _    _     _    _
#    | y[t] |   |   I        0   |   | y[t] |   |   0  |
#    |      | = |                | x |      | + |      |
#	 | x[t] |   | L[y,t]  L[f,t] |   | f[t] |   | e[t] |
#     -    -     -              -     -    -     -    -
#	 
#     _    _              _      _
#    | y[t] |            | y[t-1] |   
#    |      | = B[t-1] x |        | + u[t]
#    | f[t] |            | f[t-1] |   
#     -    -              -      -     
# where L[t] = (L[y,t] ; L[f,t]) and B[t] are coefficients, f[t] are factors, e[t]~N(0,V[t])
# and u[t]~N(0,Q[t]), and
# 
#   L[t] = L[t-1] + v[t]
#   B[t] = B[t-1] + n[t]
#
# with v[t]~N(0,H[t]), n[t]~N(0,W[t])
#
# All covariances follow EWMA models of the form:
#
#  V[t] = l_1 V[t-1] + (1 - l_1) e[t-1]e[t-1]'
#  Q[t] = l_2 Q[t-1] + (1 - l_2) u[t-1]u[t-1]'
#
# with l_1, l_2, l_3 and l_4 being the decay/forgetting factors (see paper for details).
#-----------------------------------------------------------------------------------------
#  - This code estimates a single model
#-----------------------------------------------------------------------------------------
# Written by Fawdy
# Universitas Gadjah Mada
# This version: 29 August, 2012
# Adapted from Dimitris Korobilis code
# Update version: 23 September, 2022
#-----------------------------------------------------------------------------------------

# Setting the data path
backup_wd = ''
main_wd = ''
wd = main_wd

#-------------------------------USER INPUT--------------------------------------
# Model specification
nfac = 1         # number of factors
nlag = 4         # number of lags of factors

# Control the amount of variation in the measurement and error variances
l_1 = 0.96       # Decay factor for measurement error variance
l_2 = 0.96       # Decay factor for factor error variance
l_3 = 0.99       # Decay factor for loadings error variance
l_4 = 0.99       # Decay factor for VAR coefficients error variance

# Select if y[t] should be included in the measurement equation (if it is
# NOT included, then the coefficient/loading L[y,t] is zero for all periods
y_true = 1       # 1: Include y[t]; 0: Do not include y[t]

# Select data set to use
select_data = 2  # 1: 81 variables; 2: 20 variables (as in DMA)

# Select subsample (NOTE: only if select_data = 1)
sample = 2       # 1: Use balanced panel of 18 variables after 1970:Q1
                 # 2: Use all data from 1959:Q1 (unbalanced)
                 # 3: Use all data from 1980:Q1 (unbalanced)

# Select transformations of the macro variables in Y
transf = 1       # 1: Use first (log) differences (only for CPI, GDP, M1)
                 # 2: Use annualized (CPI & GDP) & second (log) differences (M1)    
                                    
# Select a subset of the 6 variables in Y                  
subset = 1       # 1: Infl.- GDP - Int. Rate (3 vars)
                 # 2: Infl. - Unempl. - Int. Rate (3 vars)
                 # 3: Infl. - Inf. Exp. - GDP - Int. Rate (4 vars)
                 # 4: Infl. - Inf. Exp. - GDP - M1 - Int. Rate (5 vars)
                 # 5: Infl. - Inf. Exp. - Unempl. - M1 - Int. Rate (5 vars)

# Impulse responses
nhor = 21        # Impulse response horizon
resp_int = 4     # Chose the equation # where the shock is to be imposed
shock_type = 1   # 1: Cholesky - no dof adjustment 
                 # 2: Residual - 1 unit (0.5 increase of interest rate)      

# Forecasting
nfore = 16       # Forecast horizon (note: forecasts are iterated)

# Plot graphs
plot_est = 1     # 1: Plot graphs of estimated factors, volatilities etc; 0: No graphs
plot_imp = 1     # 1: Plot graphs of impulse responses; 0: No grpahs
plot_fore = 1    # 1: Plot graphs of impulse responses; 0: No grpahs

# LOAD DATA
xdata_all_xl = pd.read_excel(wd+'/xdata_all.xlsx', header=None)
xdata_xl = pd.read_excel(wd+'/xdata.xlsx', header=None)
ydata_xl = pd.read_excel(wd+'/ydata.xlsx', header=None)
ydata2_xl = pd.read_excel(wd+'/ydata2.xlsx', header=None)

xdata_all = xdata_all_xl.to_numpy(dtype=float)
xdata = xdata_xl.to_numpy(dtype=float)
ydata = ydata_xl.to_numpy(dtype=float)
ydata2 = ydata2_xl.to_numpy(dtype=float)


#----------------------------------LOAD DATA----------------------------------------
# Load Koop and Korobilis (2012) quarterly data
# load data used to extract factors
# load xdata_all.dat;
# load xdata.dat;
# load data on inflation, gdp and the interest rate 
# load ydata.dat;
# load ydata2.dat;
# load transformation codes (see file transx.m)
# load tcode.dat;
# load the file with the dates of the data (quarters)
# load yearlab.mat;


if transf == 2:
    ydata = ydata2

if select_data == 1:
    xdata = xdata_all
    # load the file with the names of the variables
    # load xnames.mat;
    if sample == 1:
        xdata = xdata[46:,[3, 4, 9, 10, 11, 14, 16, 22, 31, 32, 33, 34, 37, 38, 40, 41, 63, 66]]
        ydata = ydata[46:,:]
        # yearlab = yearlab(np.arange(46,end()+1))
        # namesXY = np.array([['Inflation'],['Infl. Exp.'],['GDP'],['Unemployemnt'],['M1'],['FedFunds'],[varnames(np.array([np.arange(3,4+1),np.arange(9,11+1),14,16,22,np.arange(31,34+1),np.arange(37,38+1),np.arange(40,41+1),63,66]))]])
    # elif sample == 2: 
        # namesXY = ['Inflation' ; 'Infl. Exp.'; 'GDP'; 'Unemployemnt'; 'M1'; 'FedFunds'; varnames ];
        # load varnames.mat;
        # namesXY = np.array([['Inflation'],['Infl. Exp.'],['GDP'],['Unemployemnt'],['M1'],['FedFunds'],[varnames]])

# Demean and standardize data (needed to extract Principal Components)
xdata = preprocessing.scale(xdata) + 1e-10
xdata[np.isnan(xdata)] = 0
ydata = preprocessing.scale(ydata)
# Select subsect of vector Y
if subset == 1:
    ydata = ydata[:,[0,2,5]]
    # namesXY = namesXY(np.array([1,3,np.arange(6,namesXY.shape[1-1]+1)]))
else:
    if subset == 2:
        ydata = ydata[:,[0,3,5]]
        # namesXY = namesXY(np.array([1,4,np.arange(6,namesXY.shape[1-1]+1)]))
    else:
        if subset == 3:
            ydata = ydata[:,[0,1,2,5]]
            # namesXY = namesXY(np.array([1,2,3,np.arange(6,namesXY.shape[1-1]+1)]))
        else:
            if subset == 4:
                ydata = ydata[:,[0,1,2,4,5]]
                # namesXY = namesXY(np.array([1,2,3,5,np.arange(6,namesXY.shape[1-1]+1)]))
            else:
                if subset == 5:
                    ydata = ydata[:,[1,2,4,5,6]]
                    # namesXY = namesXY(np.array([1,2,4,5,np.arange(6,namesXY.shape[1-1]+1)]))

# Define X and Y matrices
X = xdata

Y = ydata

# Set dimensions of useful quantities
t = Y.shape[1-1]

n = X.shape[2-1]

p = Y.shape[2-1]

r = nfac + p

q = n + p

m = nlag * (r ** 2)

k = nlag * r

# Just a small check to avoid error in input
if resp_int > r:
    print(np.array([f'Your VAR is of size {r} and you imposing a shock in equation {resp_int} !!!']))
    raise Exception('Check again your input in "resp_int"')


# =========================| PRIORS |================================
# Initial condition on the factors
factor_0 = {
    'mean': np.zeros((k,1)),
    'var' : 10 * np.eye(k)
    }
# Initial condition on lambda_t
lambda_0 = {
    'mean' : np.zeros((q,r)),
    'var' : 1 * np.eye(r)
    }
# Initial condition on beta_t
b_prior,Vb_prior = Minn_prior_KOOP(0.1,r,nlag,m)

beta_0 = {
    'mean' : b_prior,
    'var' : Vb_prior,
    }

# Initial condition on the covariance matrices
V_0 = 0.1 * np.eye(q)
V_0[0:p,0:p] = 0
Q_0 = 0.1 * np.eye(r)
# Put all decay/forgetting factors together in a vector
l = np.array([[l_1],[l_2],[l_3],[l_4]])
# Initialize impulse response analysis
impulses = np.zeros((t,q,nhor))
if resp_int <= p:
    scale = np.std(Y[:,resp_int])
else:
    scale = 1

bigj = np.zeros((r,r * nlag))
eye_r = np.eye(r)
bigj[0:r,0:r] = np.eye(r)
# Initialize matrix of forecasts
y_fore = np.zeros((nfore,r))

# =========================| END OF PRELIMENARIES |================================
# Get PC estimate using xdata up to time t

X_st = preprocessing.scale(xdata[:,:])
X_st[np.isnan(X_st)] = 0
FPC2,LPC = extract(X_st,nfac)
FPC = np.concatenate([Y, FPC2], axis=1)
YX = np.concatenate([Y,X_st], axis = 1)

L_OLS,B_OLS,beta_OLS,SIGMA_OLS,Q_OLS = ols_pc_dfm(YX,FPC,LPC,y_true,n,p,r,nfac,nlag)

# 1/ Estimate the FCI using Principal Component:
FPCA = FPC2

# 2/ Estimate the FCI using the method by Doz, Giannone and Reichlin (2011):
B_doz = np.array([[np.transpose(beta_OLS)],[np.eye(r * (nlag - 1)),np.zeros((r * (nlag - 1),r))]])
Q_doz = np.array([[Q_OLS,np.zeros((r,r * (nlag - 1)))],[np.zeros((r * (nlag - 1),k))]])
Fdraw = Kalman_companion(YX,0 * np.ones((k,1)),10 * np.eye(k),L_OLS,(SIGMA_OLS + 1e-10 * np.eye(q)),B_doz,Q_doz) # titik terakhir error
FDOZ = Fdraw

# 3/ Estimate the FCI using the method in Koop and Korobilis (2013):
# ====| STEP 1: Update Parameters Conditional on PC
beta_t,beta_new,lambda_t,V_t,Q_t = KFS_parameters(YX,FPC,l,nfac,nlag,y_true,k,m,p,q,r,t,lambda_0,beta_0,V_0,Q_0)

# ====| STEP 1: Update Factors Conditional on TV-Parameters
factor_new,Sf_t_new = KFS_factors(YX,lambda_t,beta_t,V_t,Q_t,nlag,k,r,q,t,factor_0)
#======================== END FAVAR ESTIMATION ========================






