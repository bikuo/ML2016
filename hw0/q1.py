import sys

line_num = int(sys.argv[1])
path = sys.argv[2]

file = open(path,'r')
lines = file.readlines()
M = []
for line in lines:
	tmp = line.strip().split(' ')
	#if len(tmp) != 11:
	#	print "mofo!"
	for i in range(len(tmp)):
		tmp[i] = float(tmp[i])
	M.append(tmp)
#print M

arr = []

for i in range(len(lines)):
	arr.append(M[i][line_num])

arr.sort()
print arr
# sort arr < 

if __name__ == '__main__':
	pass
