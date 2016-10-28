import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import scipy
from random import shuffle
import pdb
from math import sqrt
from scipy.special import expit
import random
np.set_printoptions(threshold='100')
import copy
## training params 
# fixed
batch = 100
feature = 57
reg = True
lamb = [0.0] if reg is False else [0.1, 0.05]
eta = [0.1]
it = 15000
# mem
w_hist = np.empty((feature, 0))
b_hist = np.empty(0)
accu_hist = np.empty(0)
eta_cp = np.copy(eta)

# changing
_weight = 2 * np.random.rand(feature, 1) -1
_weight_copy = np.copy(_weight)
_b = 0.
_b_copy = copy.deepcopy(_b)
_G_w = np.ones((feature, 1))
_G_b = 0.

##################
def normalize(A):
	for i in range(A.shape[1]):
		m = np.mean(A[:,i])
		s = np.std(A[:,i])
		A[:,i] = (A[:,i] - m) / s
	return A
def reset_params():
	global _weight, _b, _G_w, _G_b, _weight_copy, _b_copy, eta, eta_cp
	_weight = np.copy(_weight_copy)
	_b = copy.deepcopy(_b_copy)
	_G_b = 0
	_G_w = np.zeros((feature,1))
	eta = np.copy(eta_cp)

def sigmoid(X):
	return 1.0/(1 + np.exp(-X)) 

def grad(X,Y,L,e):
	global _weight,_b
	out = expit(np.dot(X,_weight) + _b)
	#loss = ((-1)*(np.dot(Y.T,np.log(out))+np.dot((1-Y).T,np.log(1-out))).sum()) / X.shape[0]
	partial_w = (-1)*np.dot(X.T,Y - out) + L * _weight
	partial_b = (-1)*(Y - out).sum()
	_weight -= e*partial_w
	_b -= e*partial_b

def evaluate(X,Y):
	global _weight, _b
	output = expit(np.dot(X,_weight)+_b)
	output = np.where(output>0.5,1,0)
	return 1 - np.sum(np.abs(output - Y))/float(Y.shape[0])

if __name__ == '__main__':
	data = pd.read_csv(sys.argv[1], header = None)
	matrix = data.as_matrix(columns = data.columns[1:])
	matrix = matrix.astype(float)
	matrix[:,:-1] = normalize(matrix[:,:-1])
	last_iter = []
	#X: input, Y: target
	X = matrix[:,:-1]
	print X[0,:]
	Y = matrix[:,-1:]
	for i in range(len(lamb)):
		for	k in range(len(eta)):
			for j in range(it):				
				#ind = np.random.randint(0,X.shape[0],size = batch)
				grad(X,Y,lamb[i],eta[k])
				eta[k] *= 0.9995
				accu = evaluate(X,Y)
				if (j+1)%1000 == 0:
					print accu,'\t#',j+1
				if j == it-1:
					last_iter = last_iter + ['lambda:'+str(lamb[i])+' eta: '+str(eta[k])+' accu: '+str(accu)]
			print "\n########################################################\n"
			w_hist = np.append(w_hist,_weight,axis = 1)
			b_hist = np.append(b_hist,_b)
			accu_hist = np.append(accu_hist,accu)
			reset_params()	
	for i in range(len(last_iter)):
		print last_iter[i]	
	max_ind = np.argmax(accu_hist)
	with open(sys.argv[2],'w') as f:
		print >>f ,' '.join(str(i) for i in w_hist[:,max_ind])
		print >>f ,b_hist[max_ind]
		