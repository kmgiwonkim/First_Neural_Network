from mnist import MNIST
import pickle
import random
import numpy as np
from pathlib import Path

print('Setting up Variables', end='')

class variables:
    
    def __init__(self, weights, biases):
        self.w = weights
        self.b = biases


eta = 0.1 #learning rate
batchSize = 15000
repetitions = 100000
n = np.array([784,32,10]) #Layers

randrange = 1

nRow = np.array([np.zeros((n[x],1)) for x in range(len(n))])#neuralRow
zRow = np.array([np.zeros((n[x],1)) for x in range(len(n))])#neuralRow pre-sigmoid

w = np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) #weights 
b = np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]) #biases

var = variables(w,b)

delta = np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]) #error
grad = np.array([ #gradient descent step
    np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]),#dC/dbias
    np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) #dC/dweight
    ])


mndata = MNIST('images')

print('.', end='')
images = np.transpose(mndata.load_training())
#images = [np.array([np.array(images[0][x]),images[1][x]]) for x in range(len(images[0]))]
print('.', end='')
mndata = MNIST('images')
imagesTe = np.transpose(mndata.load_testing())
#imagesTe = [np.array([np.array(imagesTe[0][x]),imagesTe[1][x]]) for x in range(len(imagesTe[0]))]
#This method was used because the array containing the pixel data has to be an np.array
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

def sigmoidLimit(val): #function to avoid overflow. WARNING: may greatly reduce accuracy
    return 700*(sigmoid(val*2/350)-0.5)

def sigmoidPrime(val):
    #if any(np.exp(val)>100000):
     #   print(np.exp(val))
    return np.exp(val)/np.square(1.0+np.exp(sigmoidLimit(val)))

def sigmoid(val):
    #print(val)
    return 1.0/(1.0+np.exp(-val))

def generateRandVars():
    wei = [np.array([randrange*round(random.random()*2-1,3) for x in range(n[x]*n[x+1])]).reshape((n[x+1],n[x])) for x in range(len(n)-1)] #weights
    bia = [np.array([randrange*round(random.random()*2-1,3) for x in range(n[x+1])]).reshape((n[x+1],1)) for x in range(len(n)-1)] #biases
    return bia,wei

def SGD(p,b,w,grad):#StochasticGradientDescent
    random.shuffle(images)
    for batch in range(p//batchSize):

        for index in range(batchSize):
            arrrr = costFunction(images[batch*batchSize+index],b,w)
            if len(arrrr) == 100:
                print('Yellow',arrrr,index,batch)
            costArr[batch][index] = arrrr
            cost[batch][index] = float(sum(0.5*costArr[batch][index]**2))
            bp = backProp(costArr[batch][index],grad,b,w)
            grad += bp
        #print(b,grad[0],'\n\n\n\n')
        b = b - eta*grad[0]/sum(n)
        w = w - eta*grad[1]/sum(n)
        #print(b)
        #print(len(w),len(w[0]),len(w[0][0]),len(w[1]),len(w[1][0]))
        grad = np.array([
            np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]),
            np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) 
            ])
    return b,w
    #print('\n\n\n###Gradient Descent Complete.###')

def backProp(cost,grad,b,w):
    #print('backprop',cost,zRow[len(n)-1])
    delta[len(n)-2] = np.transpose([cost])*sigmoidPrime(zRow[len(n)-1])
    #print(delta[len(n)-2])
    
    for x in range(len(n))[len(n)-3::-1]:
        delta[x] = np.matmul(np.transpose(w[x+1]),delta[x+1])*sigmoidPrime(zRow[x+1])
    grad[0] = delta
    for x in range(len(grad[1])):
        grad[1][x] = np.ravel(nRow[x])*np.transpose([np.ravel(delta[x])])
    return grad


def costFunction(img,b,w):
    nRow[0] = np.array(img[0])/255.0
    for x in range(len(n)-1):
        m = np.matmul(w[x],nRow[x])
        #print('cost',np.transpose([np.ravel(m)]),b[x])
        zRow[x+1] = np.transpose([m])+b[x]
        nRow[x+1] = sigmoid(zRow[x+1])
    nRow[len(n)-1] = np.ravel(nRow[len(n)-1])
    nRow[len(n)-1][img[1]] = 1.0 - nRow[len(n)-1][img[1]]
    return nRow[len(n)-1]


def testAveCost(b,w):
    #print(imagesTe[0],b,w)
    for index in range(len(imagesTe)):
        costArrTe[index] = 0.5*costFunction(imagesTe[index],b,w)**2.0
        costTe[index] = float(sum(costArrTe[index]))
        '''for x in costArrTe[index]:
            #print(x)
            costTe[index] += float(x**2)'''
    return sum(costTe)/float(len(costTe))


def trainVariables(Vname = 'Variables'):
    var.b,var.w = pickle.load(open('{}.p'.format(Vname),'rb'))
    print('Imported "{}.p"'.format(Vname))
    print('Calculating Initial Cost:')
    aveCost = testAveCost(var.b,var.w)
    print('Cost = {0}\n'.format(aveCost))
    prevCost = aveCost
    for x in range(repetitions):
        print('Gradient Descent {0:3}'.format(x+1), end='')
        var.b,var.w = SGD(60000,var.b,var.w,grad) #Stochastic Gradient Descent
        pickle.dump([var.b,var.w], open('{}.p'.format(Vname),'wb'))
        var.b,var.w = pickle.load(open('{}.p'.format(Vname),'rb'))
        #print('Gradient Descent',x+1,(
        print(' Complete.', end='')
        costArrTe = np.array([np.zeros(10) for x in range(len(imagesTe))])
        costTe = np.zeros(len(imagesTe))
        aveCost = testAveCost(var.b,var.w)
        print(
            ' Cost = {0:16.14f} Cost Drop = {1}.'.format(
                aveCost,prevCost - aveCost)
            )
        prevCost = aveCost


def loadUI():
    choice = None
    while(choice != 'q'):
        print('\n\n 1 = Train Variables')
        print(' 2 = Save Current Variables as...')
        print(' 3 = Load Varible File...')
        print(' 4 = Test Current Variables')
        print(' 5 = Generate New Random Variables')
        print(' q = Quit UI')
        print('\nType Desired Function:',end = '')
        choice = input()
        
        if choice not in ['1','2','3','4','5','q']:
            print('Please do your shit properly')
            
        elif choice == '1':
            print('    Training Variables.')
            print('    Input file name (0 for main variables):',end = '')
            name = input()
            if name == '0':
                trainVariables()
            else:
                my_file = Path(name)
                if my_file.is_file():
                    trainVariables(name)
                else:
                    print('    get your shit together')
            
                
        elif choice == '2':
            print('    Saving Current Variables.')
            print('    Input file name (0 for main variables):',end = '')
            name = input()
            if name == '0':
                pickle.dump([var.b,var.w], open('Variables.p','wb'))
                print('        Saved to Variables.p.')
            else:
                pickle.dump([var.b,var.w], open('{}.p'.format(name),'wb'))
                print('        Saved as {}.p.'.format(name))
            
    
        elif choice == '3':
            print('Loading Varible File.')
            print('Input file name (0 for main variables):',end = '')
            name = input()
            if name == '0':
                var.b,var.w = pickle.load(open('Variables.p','rb'))
            else:
                my_file = Path(name)
                if my_file.is_file():
                    var.b,var.w = pickle.load(open('{}.p'.format(name),'rb'))
                else:
                    print('    get your shit together')
        
        elif choice == '4':
            print('    Test Current Variables')
            random.shuffle(imagesTe)
            Cc = 0.5*costFunction(imagesTe[0],var.b,var.w)**2
            print('    Cost   =',sum(Cc))
            print('    Number =',imagesTe[0][1])
            print('    Array  =\n',Cc)
    
        elif choice == '5':
            var.b,var.w = generateRandVars()
            print('    Generated New Random Variables.')
    
    
loadUI()




