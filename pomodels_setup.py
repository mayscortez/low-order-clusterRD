import numpy as np

#########################################
# Linear (in z) Potential Outcomes Models
#########################################
linear_pom = lambda C,alpha, z : C.dot(z) + alpha

#lin_additive1

#lin_additive2

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