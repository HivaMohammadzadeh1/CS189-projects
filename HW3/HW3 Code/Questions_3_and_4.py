import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

####### QUESTION 3: Isocontours of Normal Distributions

## Part 1: 
np.random.seed(288)
f = multivariate_normal([1,1], [[1,0],[0,2]])
x_domain, y_domain = np.mgrid[-3:3:.01, -3:3:.01]
axises = np.dstack((x_domain, y_domain))
plt.figure(0)
plt.title("3-1")
plt.contourf(x_domain, y_domain, f.pdf(axises))   
plt.colorbar()
plt.show()


## Part 2: 

f = multivariate_normal([-1,2], [[2,1],[1,4]])
x_domain, y_domain = np.mgrid[-3:3:.01, -3:3:.01]
axises = np.dstack((x_domain, y_domain))
plt.figure(1)
plt.title("3-2")
plt.contourf(x_domain, y_domain, f.pdf(axises))   
plt.colorbar()
plt.show()

## Part 3: 

f1 = multivariate_normal([0,2], [[2,1],[1,1]])
f2 = multivariate_normal([2,0], [[2,1],[1,1]])
x_domain, y_domain = np.mgrid[-3:3:.01, -3:3:.01]
axises = np.dstack((x_domain, y_domain))
plt.figure(2)
plt.title("3-3")
plt.contourf(x_domain, y_domain, f1.pdf(axises) - f2.pdf(axises), 20)  
plt.colorbar()
plt.show()

## Part 4: 

f1 = multivariate_normal([0,2], [[2,1],[1,1]])
f2 = multivariate_normal([2,0], [[2,1],[1,4]])
x_domain, y_domain = np.mgrid[-3:3:.01, -3:3:.01]
axises = np.dstack((x_domain, y_domain))
plt.figure(3)
plt.title("3-4")
plt.contourf(x_domain, y_domain, f1.pdf(axises) - f2.pdf(axises), 20)  
plt.colorbar()
plt.show()

## Part 5: 

f1 = multivariate_normal([1,1], [[2,0],[0,1]])
f2 = multivariate_normal([-1,-1], [[2,1],[1,2]])
x_domain, y_domain = np.mgrid[-3:3:.01, -3:3:.01]
axises = np.dstack((x_domain, y_domain))
plt.figure(4)
plt.title("3-5")
plt.contourf(x_domain, y_domain, f1.pdf(axises) - f2.pdf(axises), 20) 
plt.colorbar()
plt.show()

#### QUESTION 4: Eigenvectors of the Gaussian Covariance Matrix

x1 = multivariate_normal.rvs(mean=3.0, cov=9.0, size=100, random_state=1)
x2 = 0.5 * x1 + multivariate_normal.rvs(mean=4.0, cov=4.0, size=100, random_state=4)
position_vectors = np.vstack((x1, x2))

## Part 1: 
mean = np.mean(position_vectors, axis=1)
print("Mean:",mean)

## Part 2: 
covariance = np.cov(position_vectors)
print("Covariance matrix:\n", covariance)

## Part 3: 
eigenvalues, eigenvectors = np.linalg.eig(covariance)
print("Eigenvectors:\n",eigenvectors)
print("Eigenvalues:\n",eigenvalues)

## Part 4: 
plt.figure(5)
plt.axis((-15,15,-15,15))
plt.title("3-4")
plt.scatter(position_vectors[0],position_vectors[1])
plt.arrow(*mean, *(eigenvalues[1]*eigenvectors[:,1]), color='green', width=0.1)
plt.arrow(*mean, *(eigenvalues[0]*eigenvectors[:,0]), color='green', width=0.1)
plt.show()

## Part 5: 
plt.figure(6)
rotated_points = eigenvectors.T @ (position_vectors.T - mean).T
plt.axis((-15,15,-15,15))
plt.title("3-5")
plt.scatter(rotated_points[0],rotated_points[1]);
plt.show()

