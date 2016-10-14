import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib as mplt
mplt.use('Qt5Agg') 
import matplotlib.pyplot as plt
import scipy
from operator import itemgetter
from random import shuffle
import pdb
from math import sqrt
np.set_printoptions(threshold='100')

## training params 
# fixed
reg = True
lamb = [0.0] if reg == False else [10.]
feature = [9]
eta = [0.1,1.,10.] # not used in ada_delta
it = 500000

_rho = 0.95
_eps = 1e-6
#mem
w_hist = np.empty((len(feature)*9,0))
b_hist = np.empty(0)
err_hist = np.empty(0)
# changing
_weight = np.random.rand(len(feature)*9,1)
_weight_copy = _weight
_b = np.random.random()
_b_copy = _b
_G_w = np.zeros((len(feature)*9,1))
_G_b = 0.

_delta_w = np.zeros((len(feature)*9,1))
_delta_b = 0.
##################
def reset_params():
	global _weight, _b, _G_w, _G_b, _weight_copy, _b_copy
	_weight = _weight_copy
	_b = _b_copy
	_G_b = 0
	_G_w = np.zeros((len(feature)*9,1))
	_delta = np.zeros((len(feature)*9,1))

def feature_normalize(X,tr_ind):
	X_norm = X
	f = len(feature)
	for i in xrange(f):
		_min = np.min(X[[tr_ind],i,:])
		_max = np.max(X[[tr_ind],i,:])
		X_norm[:,i,:] = (X_norm[:,i,:] - _min)/(_max - _min)
	return X_norm	

def ada_grad(X, Y, L, e):
	global _weight, _b, _G_w, _G_b
	
	err = np.dot(X.T,_weight) + _b - Y.T
	sq_err = np.dot(err.T , err) + L * (_weight**2).sum()
	
	partial_w = 2 * (np.dot(X,err) + _weight * L ) /err.shape[0]
	partial_b = 2 * (err.sum()) / err.shape[0]
	
	_G_w += (partial_w ** 2)
	_G_b += (partial_b ** 2)
	
	_weight = _weight - (e * partial_w) /(_G_w ** 0.5)
	_b = _b - (e * partial_b) /(_G_b**0.5)

	return  sq_err / float(err.shape[0])

def ada_delta(X, Y, L):
	global _rho, _eps, _weight, _b, _delta_w, _delta_b ,_G_w, _G_b 
	
	err = np.dot(X.T,_weight) + _b - Y.T
	sq_err = np.dot(err.T, err) + L * (_weight**2).sum()

	partial_w = 2 * (np.dot(X,err) + _weight * L) / err.shape[0]
	partial_b = 2 * (err.sum()) / err.shape[0]

	_G_w = _rho*_G_w + (1 - _rho)*partial_w ** 2
	_G_b = _rho*_G_b + (1 - _rho)*partial_b ** 2

	dw = (((_delta_w + _eps)**0.5/(_G_w + _eps)**0.5)*partial_w)
	db = (sqrt(_delta_b + _eps)/sqrt(_G_b + _eps))*partial_b

	_weight = _weight - dw
	_b = _b - db

	_delta_w = (_rho * _delta_w) + (1 - _rho) * (dw * dw)
	_delta_b = (_rho * _delta_b) + (1 - _rho) * (db * db)

	return sq_err / float(err.shape[0])

def evaluate(X,Y):
	global _weight, _b
	output = np.dot(X.T, _weight ) + _b
	sq_err = np.dot((Y.T - output).T,(Y.T - output))
	return  sq_err / float(Y.shape[1])

if __name__ == '__main__':
#read training data into m
	data = pd.read_csv('train.csv')
	a = data.as_matrix(columns = data.columns[3:])
	l = int(a.shape[0] / 18)
	new = np.empty((18,0))
	matrix = np.empty((0, 18, 480))
	for i in range(l):
		new = np.append(new,a[i*18:i*18+18,:], axis = 1)
		if (i+1) % 20 == 0:
			matrix = np.append(matrix,new.reshape(1,18,-1),axis = 0)
			new = np.zeros((18, 0))
	x = np.arange(0,12)
	shuffle(x)
	eval_ind = x[:1]
	tr_ind = x[1:]
	print tr_ind
	pm2_5 = feature.index(9)
	matrix = matrix[:,feature,:]
	matrix = matrix.astype(float)
	#matrix = feature_normalize(matrix,[tr_ind])
	monthly = 480 - 9
	data_size = monthly*len(tr_ind)
	last_iter = []
	#X: input, Y: target, Ex: evalset, Ey: evaltarget 
	X = np.empty((len(feature)*9,0))
	Y = np.empty((1,0))
	Ex = np.empty((len(feature)*9,0))
	Ey = np.empty((1,0))
	#training data
	for i in range(len(tr_ind)):
		for j in range(monthly):
			single_data = np.array([])
			for k in range(9):
				single_data =  np.append(single_data,(matrix[i,:,j+k]))
			single_data = np.reshape(single_data,(-1,1))
			X = np.append(X, single_data ,axis = 1)  
			Y = np.append(Y,matrix[i,pm2_5,j+9].reshape(1,1),axis = 1)
	pdb.set_trace()
	#validation data
	for i in range(len(eval_ind)):
		for j in range(monthly):
			a = np.array([])
			for k in range(9):
				a = np.append(a,(matrix[i,:,j+k]))
			a = np.reshape(a,(-1,1))
			Ex = np.append(Ex, a, axis = 1)
			Ey = np.append(Ey, matrix[i][pm2_5][j+9].reshape(1,1), axis = 1)
	file = open('params.mdl','w')
	
	for i in range(len(lamb)):
		for	k in range(len(eta)):
			ret1 = 0
			ret2 = 0
			for j in range(it):
				ret1 = ada_grad(X,Y,lamb[i],eta[k])
				#ret1 = ada_delta(X,Y,lamb[i])
				ret2 = evaluate(Ex, Ey)
				if (j+1)%1000 == 0:
					print ret1,'--',ret2,'\t#',j+1
				if j == it-1:
					last_iter = last_iter + ['lambda: '+str(lamb[i])+'eta: '+str(eta[k])+' err: '+str(ret1)+'-'+str(ret2)]
			print >> file,'lambda: ',lamb[i],'eta: ',eta[k],' err: ',ret1,';',ret2,'\n',_weight.T,'\n', _b
			print "\n########################################################\n"
			w_hist = np.append(w_hist,_weight,axis = 1)
			b_hist = np.append(b_hist,_b)
			err_hist = np.append(err_hist,ret2)
			reset_params()	
	for i in range(len(last_iter)):
		print last_iter[i]
	#testing
	tst = pd.read_csv('test_X.csv',header = None)
	mat = tst.as_matrix(columns = tst.columns[2:])
	mat = np.reshape(mat,(240,18,9))
	mat = mat[:,[feature],:]
	mat = mat.astype(float)
	ans = np.array([['id','value']])
	min_ind = np.argmin(err_hist)
	print w_hist[:,min_ind]
	#output csv
	for i in range(240):
		y = np.dot(mat[i,:,:].flatten() , w_hist[:,min_ind]) + b_hist[min_ind]
		index = 'id_'+str(i)
		ans = np.append(ans,np.array([[index,y]]),axis = 0)
	np.savetxt(sys.argv[1],ans, delimiter = ',',fmt = "%s")
