import numpy as np
from scipy import interpolate, special

########################################
# Estimators - SNIPE based
########################################
def SNIPE_linear(n, p, y, A, z):
    '''
    Returns an estimate of the TTE using our proposed estimator under unit RD

    n (int): number of individuals
    p (float): treatment probability
    y (numpy array?): observations
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    zz = z/p - (1-z)/(1-p)
    return 1/n * y.dot(A.dot(zz))

def SNIPE_beta(n, p, y, A, z, beta):
    # n = z.size
    # z = z.reshape((n,1))
    treated_neighb = A.dot(z)
    control_neighb = A.dot(1-z)
    est = 0
    for i in range(n):
        w = 0
        a_lim = min(beta,int(treated_neighb[i]))
        for a in range(a_lim+1):
            b_lim = min(beta - a,int(control_neighb[i]))
            for b in range(b_lim+1):
                w = w + ((1-p)**(a+b) - (-p)**(a+b)) * p**(-a) * (p-1)**(-b) * special.binom(treated_neighb[i],a)  * special.binom(control_neighb[i],b)
        est = est + y[i]*w

    return est/n

def CRD_SNIPE_linear_old(n, p, y, A, z, clusters=np.array([])):
    '''
    TODO: look at this and see if its useful at all... if not delete

    n (int): number of individuals
    p (float): treatment probability
    y (np array): outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    cluster (numpy array): TODO
    '''
    if clusters.size == 0:
        z_c = z
        A_c = A
    else:
        z_c = 1*(np.sum(np.multiply(z.reshape((n,1)),clusters),axis=0)>0)
        A_c = 1*(A.dot(clusters) >0)
    zz = z_c/p - (1-z_c)/(1-p)
    return 1/n * y.dot(A_c.dot(zz))
    
def CRD_SNIPE_linear():
    pass

def CRD_SNIPE_beta():
  pass

def CRD_SNIPE_naive_linear():
    pass

def CRD_SNIPE_naive_beta():
  pass

#######################################
# Estimators - Horvitz-Thomson & Hajek
#######################################

def horvitz_thompson(n, p, y, A, z, clusters=np.array([])):
    '''
    TODO
    '''
    if clusters.size == 0:
        zz = np.prod(np.tile(z/p,(n,1)),axis=1, where=A==1) - np.prod(np.tile((1-z)/(1-p),(n,1)),axis=1, where=A==1)
    else:
        deg = np.sum(clusters,axis=1)
        wt_T = np.power(p,deg)
        wt_C = np.power(1-p,deg)
        zz = np.multiply(np.prod(A*z,axis=1),wt_T) - np.multiply(np.prod(A*(1-z),axis=1),wt_C)
    return 1/n * y.dot(zz)

def hajek(n, p, y, A, z, clusters=np.array([])): 
    '''
    TODO
    '''
    if clusters.size == 0:
        zz_T = np.prod(np.tile(z/p,(n,1)), axis=1, where=A==1)
        zz_C = np.prod(np.tile((1-z)/(1-p),(n,1)), axis=1, where=A==1)
    else:
        deg = np.sum(clusters,axis=1)
        wt_T = np.power(p,deg)
        wt_C = np.power(1-p,deg)
        zz_T = np.multiply(np.prod(A*z,axis=1),wt_T) 
        zz_C = np.multiply(np.prod(A*(1-z),axis=1),wt_C)
    all_ones = np.ones(n)
    est_T = 0
    est_C=0
    if all_ones.dot(zz_T) > 0:
        est_T = y.dot(zz_T) / all_ones.dot(zz_T)
    if all_ones.dot(zz_C) > 0:
        est_C = y.dot(zz_C) / all_ones.dot(zz_C)
    return est_T - est_C

#################################
# Estimators - Staggered Rollout
#################################
def graph_agnostic(n, sums, H):
    '''
    Returns an estimate of the TTE with (beta+1) staggered rollout design

    n (int): popluation size
    H (numpy array): PPOM coefficients h_t or l_t
    sums (numpy array): sums of outcomes at each time step
    '''
    return (1/n)*H.dot(sums)

def poly_interp_splines(n, P, sums, spltyp = 'quadratic'):
  '''
  Returns estimate of TTE using spline polynomial interpolation 
  via scipy.interpolate.interp1d

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  spltyp (str): type of spline, can be 'quadratic, or 'cubic'
  '''
  assert spltyp in ['quadratic', 'cubic'], "spltyp must be 'quadratic', or 'cubic'"
  f_spl = interpolate.interp1d(P, sums, kind=spltyp, fill_value='extrapolate')
  TTE_hat = (1/n)*(f_spl(1) - f_spl(0))
  return TTE_hat

def poly_interp_linear(n, P, sums):
  '''
  Returns two estimates of TTE using linear polynomial interpolation 
  via scipy.interpolate.interp1d
  - the first is with kind = 'linear' (as in... ?)
  - the second is with kind = 'slinear' (as in linear spline)

  n (int): popluation size
  P (numpy array): sequence of probabilities p_t
  sums (numpy array): sums of outcomes at each time step
  '''

  #f_lin = interpolate.interp1d(P, sums, fill_value='extrapolate')
  f_spl = interpolate.interp1d(P, sums, kind='slinear', fill_value='extrapolate')
  #TTE_hat1 = (1/n)*(f_lin(1) - f_lin(0))
  TTE_hat2 = (1/n)*(f_spl(1) - f_spl(0))
  #return TTE_hat1, TTE_hat2
  return TTE_hat2

##########################
# Estimators - Regression
##########################

def est_ols(y, A, z):
    '''
    Returns an estimate of the TTE using OLS (regresses over proportion of neighbors treated)
    Uses numpy.linalg.lstsq without the use of the normal equations; assume linear model

    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    n = A.shape[0]
    X = np.ones((n,3))
    X[:,1] = z
    X[:,2] = (A.dot(z) - z) / (np.array(A.sum(axis=1)).flatten()-1+1e-10)

    v = np.linalg.lstsq(X,y,rcond=None)[0] # solve for v in y = Xv
    return v[1]+v[2]

def est_ols_treated(y, A, z):
    '''
    Returns an estimate of the TTE using OLS (regresses over number neighbors treated)
    Uses numpy.linalg.lstsq without the use of the normal equations; assumes linear model

    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    '''
    n = A.shape[0]
    X = np.ones((n,3))
    X[:,1] = z
    X[:,2] = A.dot(z) - z

    v = np.linalg.lstsq(X,y,rcond=None)[0] # solve for v in y = Xv
    return v[1]+(v[2]*(np.sum(A)-n)/n)

def poly_regression_prop(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)/(np.array(A.sum(axis=1)).flatten()-1+1e-10)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  v = np.linalg.lstsq(X,y,rcond=None)[0]
  return np.sum(v)-v[0]

def poly_regression_num(beta, y, A, z):
  '''
  Returns an estimate of the TTE using polynomial regression using
  numpy.linalg.lstsq

  beta (int): degree of polynomial
  y (numpy array): observed outcomes
  A (square numpy array): network adjacency matrix
  z (numpy array): treatment vector
  '''
  n = A.shape[0]

  if beta == 0:
      X = np.ones((n,2))
      X[:,1] = z
  else:
      X = np.ones((n,2*beta+1))
      count = 1
      treated_neighb = (A.dot(z)-z)
      for i in range(beta):
          X[:,count] = np.multiply(z,np.power(treated_neighb,i))
          X[:,count+1] = np.power(treated_neighb,i+1)
          count += 2

  # least squares regression
  v = np.linalg.lstsq(X,y,rcond=None)[0]

  # Estimate TTE
  count = 1
  treated_neighb = np.array(A.sum(axis=1)).flatten()-1
  for i in range(beta):
      X[:,count] = np.power(treated_neighb,i)
      X[:,count+1] = np.power(treated_neighb,i+1)
      count += 2
  TTE_hat = np.sum((X @ v) - v[0])/n
  return TTE_hat

###################################
# Estimators - Difference-in-means
###################################

def diff_in_means_naive(y, z):
    '''
    Returns an estimate of the TTE using difference in means
    (mean outcome of individuals in treatment) - (mean outcome of individuals in control)

    y (numpy array): observed outcomes
    z (numpy array): treatment vector
    '''
    return y.dot(z)/np.sum(z) - y.dot(1-z)/np.sum(1-z)

def diff_in_means_fraction(n, y, A, z, tol):
    '''
    Returns an estimate of the TTE using weighted difference in means where 
    we only count neighborhoods with at least tol fraction of the neighborhood being
    assigned to treatment or control

    n (int): number of individuals
    y (numpy array): observed outcomes
    A (square numpy array): network adjacency matrix
    z (numpy array): treatment vector
    tol (float): neighborhood fraction treatment/control "threshhold"
    '''
    z = np.reshape(z,(n,1))
    treated = 1*(A.dot(z)-1 >= tol*(A.dot(np.ones((n,1)))-1))
    treated = np.multiply(treated,z).flatten()
    control = 1*(A.dot(1-z)-1 >= tol*(A.dot(np.ones((n,1)))-1))
    control = np.multiply(control,1-z).flatten()

    est = 0
    if np.sum(treated) > 0:
        est = est + y.dot(treated)/np.sum(treated)
    if np.sum(control) > 0:
        est = est - y.dot(control)/np.sum(control)
    return est