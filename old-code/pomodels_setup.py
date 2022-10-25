import numpy as np

#########################################
# Linear (in z) Potential Outcomes Models
#########################################
linear_pom = lambda C,alph,z : C.dot(z) + alph

# Linear Additive Model I from Gui, Xu, Bhasin, Han (2015) paper
linear_add1 = lambda alph,beta,gam,A,z : alph + beta*z + gam*((A.dot(z) - z) / (np.array(A.sum(axis=1)).flatten()-1+1e-10))

def lin_additive2(alp0, alp1, gam0, gam1, A, z):
    '''
    Linear Additive Model II from Gui, Xu, Bhasin, Han (2015) paper

    alp0,alp1 (float): baseline effect if unit i is not treated / treated
    gam0,gam1 (float): network effect if unit is is not treated / treated
    A (scipy sparse matrix): adjacency matrix of the network
    z (numpy array): treatment vector
    '''
    frac_treated = (A.dot(z) - z) / (np.array(A.sum(axis=1)).flatten()-1+1e-10)
    indicator0 = (z==0)*1
    indicator1 = (z==1)*1
    lam2 = (indicator0 * (alp0 + gam0*frac_treated)) + (indicator1 * (alp1 + gam1*frac_treated))
    return lam2

#############################################
# Polynomial (in z) Potential Outcomes Models
#############################################

# Scale the effects of higher order terms
a1 = 1      # for linear effects
a2 = 1      # for quadratic effects
a3 = 1      # for cubic effects
a4 = 1      # for quartic effects

# Define f(z)
f_linear = lambda alpha, z, gz: alpha + a1*z
f_quadratic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz)
f_cubic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3)
f_quartic = lambda alpha, z, gz: alpha + a1*z + a2*np.multiply(gz,gz) + a3*np.power(gz,3) + a4*np.power(gz,4)

def ppom(beta, C, alpha):
  '''
  Returns beta-degree polynomial potential outcomes function fy for beta in {0,1,2,3,4}
  
  f (function): must be of the form f(z) = alpha + z + a2*z^2 + a3*z^3 + ... + ak*z^beta
  C (np.array): weighted adjacency matrix
  alpha (np.array): vector of null effects
  '''

  if beta == 0:
      return lambda z: alpha + a1*z
  elif beta == 1:
      f = f_linear
      return lambda z: alpha + a1*C.dot(z)
  else:
      g = lambda z : C.dot(z) / np.array(np.sum(C,1)).flatten()
      if beta == 2:
          f = f_quadratic
      elif beta == 3:
          f = f_cubic
      elif beta == 4:
          f = f_quartic
      else:
          print("ERROR: invalid degree")
      return lambda z: f(alpha, C.dot(z), g(z)) 