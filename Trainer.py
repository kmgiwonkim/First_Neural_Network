from mnist import MNIST
import random
import numpy as np


print('Setting up Variables', end='')
n = np.array([784,16,16,10])
nRow = np.array([np.zeros((n[x],1)) for x in range(len(n))])#neuralRow
zRow = np.array([np.zeros((n[x],1)) for x in range(len(n))])#neuralRow pre-sigmoid
w = np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) #weights 
b = np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]) #biases
delta = np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]) #error
grad = np.array([ #gradient descent step
    np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]),#dC/dbias
    np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) #dC/dweight
    ])
eta = 0.009 #learning rate
batchSize = 15000
repetitions = 100000

v = {0:b,1:w}
mndata = MNIST('images')
images = mndata.load_training()
print('.', end='')
images = [np.array([np.array(images[0][x]),images[1][x]]) for x in range(len(images[0]))]
print('.', end='')
mndata = MNIST('images')
imagesTe = mndata.load_testing()
imagesTe = [np.array([np.array(imagesTe[0][x]),imagesTe[1][x]]) for x in range(len(imagesTe[0]))]
print('.', end='')
costArr = np.array([
    [np.zeros(n[len(n)-1]) for x in range(batchSize)]
    for x in range(len(images)//batchSize)
    ])
cost = np.array([
    np.zeros(batchSize) for x in range(len(images)//batchSize)
    ])

costArrTe = np.array([np.zeros(n[len(n)-1]) for x in range(len(imagesTe))])
costTe = np.zeros(len(imagesTe))
aveCost = 0
prevCost = 0
print(' Complete.')

def sigmoidLimit(val):
    return 700*(sigmoid(val*2/350)-0.5)

def sigmoidPrime(val):
    #if any(np.exp(val)>100000):
     #   print(np.exp(val))
    return np.exp(val)/np.square(1.0+np.exp(sigmoidLimit(val)))

def sigmoid(val):
    #print(val)
    return 1.0/(1.0+np.exp(-val))

def SGD(p,b,w,grad):#StochasticGradientDescent
    random.shuffle(images)
    for batch in range(p//batchSize):

        '''
        if batch == 0:
            print("Initializing Batch", batch+1, "/", len(range(p//batchSize)))
        elif (batch+1)%(10) == 0:
            print("Initializing Batch", batch+1, "/", len(range(p//batchSize)))
        elif batch == p//batchSize:
            print("Initializing Batch", batch+1, "/", len(range(p//batchSize)))
        '''
        for index in range(batchSize):
            '''print("   Image",(index+1),'/',batchSize, "of Batch",batch+1
                  , "/", len(range(p//batchSize)))
            '''
            costArr[batch][index] = costFunction(images[batch*batchSize+index],b,w)
            for x in costArr[batch][index]:
                cost[batch][index] += float(x**2)
            #print('yow',cost[batch][index])
            #print(costArr[batch][index])
            bp = backProp(costArr[batch][index],grad,b,w)
            #print(bp)
            grad += bp
            #print(grad)
        b = b - eta*grad[0]/sum(n)
        w = w - eta*grad[1]/sum(n)
        grad = np.array([
            np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]),
            np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) 
            ])
    return b,w
    #print('\n\n\n###Gradient Descent Complete.###')

def backProp(cost,grad,b,w):
    delta[len(n)-2] = np.reshape(cost*sigmoidPrime(np.ravel(zRow[len(n)-1])),(n[len(n)-1],1))
    
    
    for x in range(len(n))[len(n)-3::-1]:
        delta[x] = np.matmul(np.transpose(w[x+1]),delta[x+1])*sigmoidPrime(zRow[x+1])
    
    #print('Delta:',delta,'\nDone')
    grad[0] = delta
    for x in range(len(grad[1])):
        grad[1][x] = np.ravel(nRow[x])*np.reshape(delta[x], (len(delta[x]), 1))
    return grad


def costFunction(img,b,w):
    nRow[0] = img[0]/255.0
    
    for x in range(3):
        m = np.matmul(w[x],nRow[x])
        m = np.reshape(np.ravel(m), (len(m), 1))
        zRow[x+1] = m+b[x]
        nRow[x+1] = sigmoid(zRow[x+1])
    
    
    nRow[3]=np.ravel(nRow[3])
    
    costFac = np.zeros(n[len(n)-1])
    costFac[img[1]] = 1.0
    #print(nRow[3])
    #print(costFac)
    return (nRow[3] - costFac)


def importVar(fileN = 'Variables'):
    fh = open(fileN,'r')
    rData = fh.read()
    fh.close()
    x = [0,0,0,0,0]
    #x[0] Determines the position to read in the file
    #x[1] Determines the Column number in matrix
    #x[2] Determines the row number in matrix
    #x[3] Determines which set of neurons it belongs to
    #x[4] Determines whether if its weights or biases

    while(x[0]+2<len(rData)):
        if rData[x[0]+2] == '*': #switch from biases to weight
            x = [x[0]+2,0,0,0,x[4]+1]
        elif rData[x[0]+2] == '@': #switch to next 'set' of neurons
            x = [x[0]+2,0,0,x[3]+1,x[4]]
        elif rData[x[0]] == '\n':
            if rData[x[0]+1] == '\n': #switch to next matrix row
                x = [x[0]+1,0,x[2]+1,x[3],x[4]]
            else:
                k = ''
                while(rData[x[0]+1] != '\n'):
                    k+=rData[x[0]+1]
                    x[0] += 1
                v[x[4]][x[3]][x[2]][x[1]] = round(float(k),6)
                x[1] += 1
        else:
            x[0]+=1


            
def rewriteVar(b,w):
    q=[b,w]
    fh = open('Variables','w')
    fh.write('')
    fh.close()
    fh = open('Variables','a+')

    for s in range(2):
        for x in range(len(q[s])):
            for j in range(len(q[s][x])):
                for k in range(len(q[s][x][j])):
                    fh.write(str(q[s][x][j][k])+'\n')
                fh.write('\n')
            fh.write('@@\n')
        fh.write('**\n')
    fh.close()

def saveVar(fileN = 'Variables'):
    fh = open(fileN,'w')
    fh.write('')
    fh.close()
    fh = open(fileN,'a+')

    for s in range(2):
        for x in range(len(v[s])):
            for j in range(len(v[s][x])):
                for k in range(len(v[s][x][j])):
                    fh.write(str(v[s][x][j][k])+'\n')
                fh.write('\n')
            fh.write('@@\n')
        fh.write('**\n')
    fh.close()

    
def testAveCost(b,w):
    #print(imagesTe[0],b,w)
    for index in range(len(imagesTe)):
        costArrTe[index] = costFunction(imagesTe[index],b,w)

        for x in costArrTe[index]:
            #print(x)
            costTe[index] += float(x**2)
    return sum(costTe)/float(len(costTe))


'''
importVar('Cost 0.901')
random.shuffle(images)
for x in range(2):
    Cc = costFunction(images[x],b,w)**2
    print(backProp(Cc,grad,b,w))
    print(x,'=',Cc,'\n\n')
'''

importVar()
print('Imported Variables.')
print('Calculating Initial Cost:')
aveCost = testAveCost(b,w)
print('Cost =', aveCost,'\n')
prevCost = aveCost
for x in range(repetitions):
    print('Gradient Descent',x+1, end='')
    b,w = SGD(60000,b,w,grad) #Stochastic Gradient Descent
    rewriteVar(b,w)
    importVar()    
    #print('Gradient Descent',x+1,(
    print(' Complete.', end='')
    costArrTe = np.array([np.zeros(10) for x in range(len(imagesTe))])
    costTe = np.zeros(len(imagesTe))
    aveCost = testAveCost(b,w)
    print(' Cost =',aveCost, end='')
    print(' Cost Drop =',prevCost - aveCost,'.')
    prevCost = aveCost

'''

importVar()
print('Cost =',testAveCost(b,w),'.\n')
fileName = 'Cost 0.906'
saveVar(fileName)
print('Variables saved as "',fileName,'"')
'''


