from mnist import MNIST
import pickle
from random import shuffle
import numpy as np
from pathlib import Path
import time

print('Setting up Variables', end='')


class variables:
    
    def __init__(self, weights, biases):
        self.w = weights
        self.b = biases


eta = 0.3 #learning rate (now independent on batchSize)
batchSize = 1
repetitions = 30
n = np.array([784,28,10]) #Layers

randrange = 1

nRow = np.array([np.zeros((n[x],1)) for x in range(len(n))])#neuralRow
zRow = np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]])#neuralRow pre-sigmoid

w1 = np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) #weights 
b1 = np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]) #biases

var = variables(w1,b1)

delta = np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]) #error
grad = np.array([ #gradient descent step
    np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]),#dC/dbias
    np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) #dC/dweight
    ])


mndata = MNIST('images')

print('.', end='')
images = mndata.load_training()
images = [np.array([np.array(images[0][x]),images[1][x]]) for x in range(len(images[0]))]
#CONSIDER USING 'zip()' INSTEAD
print('.', end='')
mndata = MNIST('images')
imagesTe = mndata.load_testing()
imagesTe = [np.array([np.array(imagesTe[0][x]),imagesTe[1][x]]) for x in range(len(imagesTe[0]))]
#This method was used because the array containing the pixel data has to be an np.array
#There seems to be a bug where 'np.transpose()' seems to make every entry of 'images' the same
#this was tested to be faster than 'np.transpose()'

print('.', end='')



costArr = np.array([
    [np.zeros(n[-1]) for x in range(batchSize)]
    for x in range(len(images)//batchSize)
    ])
cost = np.array([
    np.zeros(batchSize) for x in range(len(images)//batchSize)
    ])
costArrTot = np.array([
    np.zeros(n[-1]) for x in range(len(images)//batchSize)
    ])

costArrTe = np.array([np.zeros(n[-1]) for x in range(len(imagesTe))])
costTe = np.zeros(len(imagesTe))
aveCost = 0
prevCost = 0
print(' Complete.')
print('Layers:',n)
print('Learning Rate:',eta)
print('Batch Size:',batchSize)

def sigmoidLimit(val): #function to avoid overflow. WARNING: may greatly reduce accuracy
    return 700*(sigmoid(val*2/350)-0.5)

def sigmoidPrime(val):
    #sigmoid(z)*(1-sigmoid(z))
    return np.array(np.exp(-val)/np.square(1.0+np.exp(-val)))

def sigmoid(val):
    return np.array(1.0/(1.0+np.exp(-val)))

def generateRandVars():#Generates a random set of weights and biases
    wei = [np.random.randn(n[x+1],n[x]) for x in range(len(n)-1)] #weights
    
    bia = [np.random.randn(n[x+1],1) for x in range(len(n)-1)] #biases
    return bia,wei

def SGD(b,w,grad):#StochasticGradientDescent
    shuffle(images)#Shuffle works properly
    for batch in range(len(images)//batchSize):
        for index in range(batchSize):
            costArr[batch][index] = costFunction(images[batch*batchSize+index],b,w)
            grad += backProp(costArr[batch][index],grad,b,w)
        b = b - eta*grad[0]
        w = w - eta*grad[1]
        grad = np.array([
            np.array([np.zeros((n[x],1)) for x in range(len(n))[1:]]),
            np.array([np.zeros((n[x],n[x-1])) for x in range(len(n))[1:]]) 
            ])
    return b,w


def backProp(cost,grad,b,w):
    delta[-1] = np.transpose([cost])*sigmoidPrime(zRow[-1])
    for x in range(len(n))[len(n)-3::-1]:
        delta[x] = np.matmul(np.transpose(w[x+1]),delta[x+1])*sigmoidPrime(zRow[x])
    grad[0] = np.array(delta)
    for x in range(len(grad[1])):
        grad[1][x] = np.array(np.ravel(nRow[x]) * delta[x])
    return grad


def costFunction(img,b,w):
    nRow[0] = np.transpose([img[0]])/255.0
    for x in range(len(n)-1):
        zRow[x] = np.matmul(w[x],nRow[x])+b[x]
        nRow[x+1] = sigmoid(zRow[x])
    nRow[-1] = np.ravel(nRow[-1])
    nRow[-1][img[1]] = nRow[-1][img[1]] - 1.0
    return nRow[-1]


def feedforward(img,b,w):
    nRow[0] = np.transpose([img[0]])/255.0
    for x in range(len(n)-1):
        zRow[x] = np.matmul(w[x],nRow[x])+b[x]
        nRow[x+1] = sigmoid(zRow[x])
    nRow[-1] = np.ravel(nRow[-1])
    return nRow[-1]


def testAveCost(b,w):
    for index in range(len(imagesTe)):
        costArrTe[index] = 0.5*costFunction(imagesTe[index],b,w)**2.0
        costTe[index] = float(sum(costArrTe[index]))
    return sum(costTe)/float(len(costTe))


def testPercentAccuracy():
    match = 0
    for x in imagesTe:
        Cc = feedforward(x,var.b,var.w)
        if x[1] == np.nonzero(Cc == max(Cc))[0][0]:
            match += 1
    return match


def trainVariables(Vname = 'Variables'):
    var.b,var.w = pickle.load(open('{}.p'.format(Vname),'rb'))
    print('\n\n\n\nImported "{0}.p"\n'.format(Vname))
    print('Layers:',n)
    print('Learning Rate:',eta)
    print('Batch Size:',batchSize)
    print('Starting % Accuracy: {0:5.2f}%'.format(testPercentAccuracy()/len(imagesTe)*100))
    for x in range(repetitions):
        print('Gradient Descent {0:3}'.format(x+1), end='')
        var.b,var.w = SGD(var.b,var.w,grad) #Stochastic Gradient Descent
        pickle.dump([var.b,var.w], open('{}.p'.format(Vname),'wb'))
        var.b,var.w = pickle.load(open('{}.p'.format(Vname),'rb'))

        if 1:
            print(' Complete:',end='')
            print(' {0:5.2f}%'.format(testPercentAccuracy()/len(imagesTe)*100),end='')
            print(' {0:7.5f}'.format(testAveCost(var.b,var.w)))
        else:
            print(' Complete.')
        


def UI():
    choice = None
    print('\n\n 1 = Train Variables')
    print(' 2 = Save Current Variables as...')
    print(' 3 = Load Varible File...')
    print(' 4 = Test Current Variables')
    print(' 5 = Generate New Random Variables')
    print(' d = Display List of Commands')
    print(' q = Quit UI')
    while(choice != 'q'):
        print('\nType Desired Function:',end = '')
        choice = input()
        
        if choice not in ['1','2','3','4','5','d','q']:
            print('Please do your shit properly')
            
        elif choice == 'd':
            print('\n\n 1 = Train Variables')
            print(' 2 = Save Current Variables as...')
            print(' 3 = Load Varible File...')
            print(' 4 = Test Current Variables')
            print(' 5 = Generate New Random Variables')
            print(' d = Display List of Commands')
            print(' q = Quit UI')
            
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
                print('        Saved to "Variables.p".')
            else:
                pickle.dump([var.b,var.w], open('{}.p'.format(name),'wb'))
                print('        Saved as {}.p.'.format(name))
            
    
        elif choice == '3':
            print('    Loading Varible File.')
            print('    Input file name (0 for main variables):',end = '')
            name = input()
            if name == '0':
                var.b,var.w = pickle.load(open('Variables.p','rb'))
            else:
                my_file = Path(name)
                if my_file.is_file():
                    var.b,var.w = pickle.load(open('{}'.format(name),'rb'))
                else:
                    print('    get your shit together')
        
        elif choice == '4':
            print('    Test Current Variables:',end='')
            print(' {0:5.2f}%'.format(testPercentAccuracy()/len(imagesTe)*100))
    
        elif choice == '5':
            var.b,var.w = generateRandVars()
            print('    Generated New Random Variables.')
    
    
UI()


