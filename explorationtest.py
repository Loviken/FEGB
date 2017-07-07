# This is a test to see that exploration works as it should.

import matplotlib.pyplot as plt
import numpy as np

# Constants
n	  = 10
dof	  = 100
alpha = 1e-6

# Create samples
X = np.zeros((n, dof))
t = np.linspace(0,1,dof)
for i in range(n):
	
	X[i] = t*(t - np.random.rand()) / (t + 1 + np.random.rand())**2
	X[i]+= 0.01*t*np.random.randn(dof)
	plt.plot(X[i])
	
mu = np.mean(X,0)
plt.plot(mu, '.', color='k')

plt.show()

cov = np.cov(X.T)
cov+= alpha*np.eye(dof)
L   = np.linalg.cholesky(cov)

#plt.imshow(cov, interpolation = 'none')
#plt.show()


# Approximate
n_aprx = 10
Approx = mu + np.matmul(np.random.randn(n_aprx, dof), L.T)

for i in range(n_aprx):
	plt.plot(Approx[i])
	
plt.plot(mu, '.', color='k')
plt.show()
