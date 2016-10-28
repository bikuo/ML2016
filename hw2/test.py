import numpy as np
import pandas as pd	
from scipy.special import expit
from sys import argv

def normalize(A):
	for i in range(A.shape[1]):
		m = np.mean(A[:,i])
		s = np.std(A[:,i])
		A[:,i] = (A[:,i] - m) / s
	return A

if __name__ == '__main__':
	#read tst file
	tst = pd.read_csv(argv[2],header = None)
	mat = tst.as_matrix(columns = tst.columns[1:])
	mat = mat.astype(float)
	mat = normalize(mat)
	ans = np.array([['id','label']])
	#read model
	md = open(argv[1],'r')
	params = md.readlines()
	weight = np.array(params[0].strip().split(' '))
	weight = weight.astype(float)
	weight = weight.reshape((-1,1))
	bias = float(params[1])
	#output csv
	for i in range(mat.shape[0]):
		y = expit(np.dot(mat[i,:], weight) + bias)
		if y >= 0.5:
			y = 1
		else:
			y = 0
		index = str(i+1)
		ans = np.append(ans,np.array([[index,y]]),axis = 0)
	np.savetxt(argv[3],ans, delimiter = ',',fmt = "%s")
