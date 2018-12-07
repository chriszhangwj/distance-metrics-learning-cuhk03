import numpy as np
a=np.array([1,2,3,4,5])
a=a.reshape(1,5)
b=np.array([3,1,6,2,3])
b=b.reshape(1,5)
c=np.vstack((a,b))
print(c)
print(c.T)
#print(a.shape)
#print(c.shape)
#print(np.cov(c.T))

#x = [-2.1, -1,  4.3]
#y = [3,  1.1,  0.12]
#X = np.stack((x, y), axis=0)
#print(X)
#print(np.cov(X))

#x = [-2.1, -1,  4.3]
#y = [3,  1.1,  0.12]
#x=np.reshape(1,3)
#y=np.reshape(1,3)
#X = np.stack((x, y), axis=0)
#print(np.cov(X))