from mnist import MNIST
import random
import numpy as np

randrange = 1

n = np.array([784,16,16,10])
w = [np.array([randrange*round(random.random()*2-1,3) for x in range(n[x]*n[x+1])]).reshape((n[x+1],n[x])) for x in range(len(n)-1)] #weights
b = [np.array([randrange*round(random.random()*2-1,3) for x in range(n[x+1])]).reshape((n[x+1],1)) for x in range(len(n)-1)] #biases
v = {0:b,1:w}


fh = open('Variables','w')
fh.write('')
fh.close()
fh = open('Variables','a+')

for s in range(2):
    for x in range(len(v[s])):
        for j in range(len(v[s][x])):
            for k in range(len(v[s][x][j])):
                fh.write(str(v[s][x][j][k])+'\n')
            fh.write('\n')
        fh.write('@@\n')
    fh.write('**\n')
fh.close

print("Weights and Biases Randomized")
