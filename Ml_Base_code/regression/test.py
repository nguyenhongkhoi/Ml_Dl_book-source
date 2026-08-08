
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180,
183]]).T  #height

y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]) #weight

#w = (X'.X)-1.(X'.y)
one = np.ones((X.shape[0], 1)) #[1,..]
Xbar = np.concatenate((one,X),  axis = 1) #X

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T,y)

w = np.dot(np.linalg.pinv(A), b)

print(w)

print("\n")
w_0, w_1 = w[0], w[1]

print(w_0)
