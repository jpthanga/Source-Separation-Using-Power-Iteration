import numpy as np
import scipy as sc
from scipy import io as io
from matplotlib import pyplot as plt
from numpy import linalg
import copy

eVectors = []
eValues = []

def powerIteration(a):
    j,k = a.shape
    v = np.matrix(np.ones((128,1)))
    for j in range(0,2):
        l= 0
        for i in range(1,100000):
            x = l
            v = a*v
            l = np.linalg.norm(v)
            v = v/l
            if abs(l-x) < 0.005:
                eVectors.append(v)
                eValues.append(l)
                a = a - l*v*np.transpose(v)
                v = np.matrix(np.ones((128,1)))
                break

def colorM(a):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.imshow(a)
    fig.colorbar(cax)
    plt.show()

def plotEigen(E):
    P = copy.deepcopy(E)
    K = P
    for i in range(0,10):
        K = np.concatenate([K,P])
    K = K.reshape((11,128))
    colorM(np.transpose(K))


flute = io.loadmat("./flute.mat")
T = np.array(flute.get('X'))

cov = np.cov(T)
powerIteration(cov)

colorM(T)
plotEigen(eVectors[0])
plotEigen(eVectors[1])

u1 = np.dot(np.transpose(cov),eVectors[0])/np.linalg.norm(np.dot(np.transpose(eVectors[0]),cov))
u2 = np.dot(np.transpose(cov),eVectors[1])/np.linalg.norm(np.dot(np.transpose(eVectors[0]),cov))
plotEigen(u1)
plotEigen(u2)

final1 = np.dot(np.transpose(eVectors[0]*eValues[0]),u1)
colorM(final1)
final2 = np.dot(np.transpose(eVectors[1]*eValues[1]),u1)
colorM(final2)





