from loader import MNIST
import pickle
from random import shuffle
import numpy as np
from pathlib import Path
import time
from PIL import Image


class variables:
    
    def setup(self,n,eta,batchSize,repetitions,Dataset):
        print('Setting up Variables', end='')
        self.n = np.array(n) #Layers
        self.eta = eta #learning rate (now independent on batchSize)
        self.batchSize = batchSize
        self.repetitions = repetitions
        self.Dataset = Dataset #DataSet Option
        
        
        self.randrange = 1
        
        self.dSetIndex = {0:'Database-MNIST',1:'Database-EMNIST'}
        
        self.w = np.array([np.zeros((self.n[x],self.n[x-1])) for x in range(len(self.n))[1:]]) #weights 
        self.b = np.array([np.zeros((self.n[x],1)) for x in range(len(self.n))[1:]]) #biases
        self.nRow = np.array([np.zeros((self.n[x],1)) for x in range(len(self.n))])#neuralRow
        self.zRow = np.array([np.zeros((self.n[x],1)) for x in range(len(self.n))[1:]])#neuralRow pre-sigmoid
        
        self.delta = np.array([np.zeros((self.n[x],1)) for x in range(len(self.n))[1:]]) #error
        self.grad = np.array([ #gradient descent step
            np.array([np.zeros((self.n[x],1)) for x in range(len(self.n))[1:]]),#dC/dbias
            np.array([np.zeros((self.n[x],self.n[x-1])) for x in range(len(self.n))[1:]]) #dC/dweight
            ])
        self.aveCost = 0
        self.prevCost = 0
        
        self.images = None
        self.imagesTe = None
        
        
    def imagesSetup(self,Dataset):
        self.Dataset = Dataset
        self.mndata = MNIST(self.dSetIndex[Dataset])
        
        if Dataset == 0: #MNIST Setup
            print('.', end='')
            images = self.mndata.load_training()
            self.images = [np.array([np.array(images[0][x]),images[1][x]]) for x in range(len(images[0]))]
            #CONSIDER USING 'zip()' INSTEAD
            print('.', end='')
            self.mndata = MNIST(self.dSetIndex[Dataset])
            imagesTe = self.mndata.load_testing()
            self.imagesTe = [np.array([np.array(imagesTe[0][x]),imagesTe[1][x]]) for x in range(len(imagesTe[0]))]
        
        elif Dataset == 1: #EMNIST Setup
            print('.', end='')
            images = self.mndata.load_training()
            self.images = [np.array([np.ravel(np.transpose([np.reshape(images[0][x],(28,28))])),images[1][x]]) for x in range(len(images[0]))]
            print('.', end='')
            self.mndata = MNIST(self.dSetIndex[Dataset])
            imagesTe = self.mndata.load_testing()
            self.imagesTe = [np.array([np.ravel(np.transpose([np.reshape(imagesTe[0][x],(28,28))])),imagesTe[1][x]]) for x in range(len(imagesTe[0]))]
            #EMNIST Database Digits image matrices were 'transposed' so had to be transposed back

        #This method was used because the array containing the pixel data has to be an np.array
        #There seems to be a bug where 'np.transpose()' seems to make every entry of 'images' the same
        #This was tested to be faster than 'np.transpose()'
        print('.', end='')
        
        self.costArr = np.array([
            [np.zeros(self.n[-1]) for x in range(self.batchSize)]
            for x in range(len(self.images)//self.batchSize)
            ])
        self.cost = np.array([
            np.zeros(self.batchSize) for x in range(len(self.images)//self.batchSize)
            ])
        self.costArrTot = np.array([
            np.zeros(self.n[-1]) for x in range(len(self.images)//self.batchSize)
            ])
        
        self.costArrTe = np.array([np.zeros(self.n[-1]) for x in range(len(self.imagesTe))])
        self.costTe = np.zeros(len(self.imagesTe))
        print(' Complete.\n')
        


def sigmoidLimit(val): #function to avoid overflow. WARNING: may greatly reduce accuracy
    return 700*(sigmoid(val*2/350)-0.5)

def sigmoidPrime(val):
    #sigmoid(z)*(1-sigmoid(z))
    return np.array(np.exp(-val)/np.square(1.0+np.exp(-val)))

def sigmoid(val):
    return np.array(1.0/(1.0+np.exp(-val)))

def generateRandVars():#Generates a random set of weights and biases
    wei = [np.random.randn(var.n[x+1],var.n[x]) for x in range(len(var.n)-1)] #weights
    bia = [np.random.randn(var.n[x+1],1) for x in range(len(var.n)-1)] #biases
    return bia,wei



def SGD(b,w,grad,eta):#StochasticGradientDescent
    shuffle(var.images)#Shuffle works properly
    for batch in range(len(var.images)//var.batchSize):
        for index in range(var.batchSize):
            var.costArr[batch][index] = costFunction(var.images[batch*var.batchSize+index],b,w)
            grad += backProp(var.costArr[batch][index],grad,b,w)
        b = b - eta*grad[0]
        w = w - eta*grad[1]
        grad = np.array([
            np.array([np.zeros((var.n[x],1)) for x in range(len(var.n))[1:]]),
            np.array([np.zeros((var.n[x],var.n[x-1])) for x in range(len(var.n))[1:]]) 
            ])
    return b,w


def backProp(cost,grad,b,w):
    var.delta[-1] = np.transpose([cost])*sigmoidPrime(var.zRow[-1])
    for x in range(len(var.n))[len(var.n)-3::-1]:
        var.delta[x] = np.matmul(np.transpose(w[x+1]),var.delta[x+1])*sigmoidPrime(var.zRow[x])
    var.grad[0] = np.array(var.delta)
    for x in range(len(var.grad[1])):
        var.grad[1][x] = np.array(np.ravel(var.nRow[x]) * var.delta[x])
    return var.grad


def costFunction(img,b,w):
    var.nRow[0] = np.transpose([img[0]])/255.0
    for x in range(len(var.n)-1):
        var.zRow[x] = np.matmul(w[x],var.nRow[x])+b[x]
        var.nRow[x+1] = sigmoid(var.zRow[x])
    var.nRow[-1] = np.ravel(var.nRow[-1])
    var.nRow[-1][img[1]] = var.nRow[-1][img[1]] - 1.0
    return var.nRow[-1]


def feedforward(img,b,w):
    var.nRow[0] = np.transpose([img[0]])/255.0
    for x in range(len(var.n)-1):
        var.zRow[x] = np.matmul(w[x],var.nRow[x])+b[x]
        var.nRow[x+1] = sigmoid(var.zRow[x])
    var.nRow[-1] = np.ravel(var.nRow[-1])
    return var.nRow[-1]


def testAveCost(b,w):
    for index in range(len(var.imagesTe)):
        var.costArrTe[index] = 0.5*costFunction(var.imagesTe[index],b,w)**2.0
        var.costTe[index] = float(sum(var.costArrTe[index]))
    return sum(var.costTe)/float(len(var.costTe))


def testPercentAccuracy():
    match = 0
    notMatch = []
    for x in var.imagesTe:
        Cc = feedforward(x,var.b,var.w)
        if x[1] == np.nonzero(Cc == max(Cc))[0][0]:
            match += 1
        else:
            notMatch.append(x)
    return match


def printParameters():
    print('Layers:',var.n)
    print('Learning Rate:',var.eta)
    print('Batch Size:',var.batchSize)
    print('Dataset:',var.dSetIndex[var.Dataset])
    print('Repetitions:',var.repetitions)
    print('Datasize\n   Training: {}\n   Test: {}'.format(len(var.images),len(var.imagesTe)))


def trainVariables(Vname = 'Variables.p'):
    var.b,var.w = pickle.load(open('{}'.format(Vname),'rb'))
    print('\n\n\n\nImported "{0}"\n'.format(Vname))
    printParameters()
    print('Starting % Accuracy: {0:5.2f}%\n\n'.format(testPercentAccuracy()/len(var.imagesTe)*100))
    for x in range(var.repetitions):
        print('Gradient Descent {0:3}'.format(x+1), end='')
        var.b,var.w = SGD(var.b,var.w,var.grad,var.eta) #Stochastic Gradient Descent
        pickle.dump([var.b,var.w], open('{}'.format(Vname),'wb'))
        var.b,var.w = pickle.load(open('{}'.format(Vname),'rb'))
        print(' Complete:',end='')
        print(' {0:5.2f}%'.format(testPercentAccuracy()/len(var.imagesTe)*100),end='')
        print(' {0:7.5f}'.format(testAveCost(var.b,var.w)))




def UI():
    printParameters()
    choice = 'd'
    while(choice != 'q'):
        
        if choice not in ['1','2','3','4','5','6','d','q']:
            print('Please do your shit properly')
            
        elif choice == 'd':
            print('\n\n 1 = Train Variables')
            print(' 2 = Save Current Variables as...')
            print(' 3 = Load Varible File...')
            print(' 4 = Test Current Variables')
            print(' 5 = Generate New Random Variables')
            print(' 6 = Setup Different Dataset')
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
                pickle.dump([var.b,var.w], open('{}'.format(name),'wb'))
                print('        Saved as {}.'.format(name))
            
    
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
            print('    Test Current Variables:')
            print('        1 = Overall Accuracy')
            print('        2 = Single Image')
            print('        3 = Load Image')
            print('        Input:',end='')
            ch = input()
            if ch not in ['1','2','3']:
                print('get your shit together')
            elif ch == '1':
                print('        {0:5.2f}%'.format(testPercentAccuracy()/len(var.imagesTe)*100))
                
                
            elif ch == '2':
                shuffle(var.imagesTe)
                Cc = feedforward(var.imagesTe[0],var.b,var.w)
                print('        Array   =\n',Cc[0:5],'\n',Cc[5:10])
                print('        Number  =',var.imagesTe[0][1])
                print('        Largest =',np.nonzero(Cc == max(Cc))[0][0])
                Cc[var.imagesTe[0][1]] = 1.0 - Cc[var.imagesTe[0][1]]
                Cc = 0.5*Cc**2.0
                print('        Cost    =',sum(Cc))
                print(var.mndata.display(var.imagesTe[0][0]))
                
            elif ch == '3':
                print('            Input Image Name:',end='')
                name = input()
                my_file = Path(name)
                if my_file.is_file():
                    Img = Image.open('{}'.format(name)).convert('1')
                    px = Img.load()
                    newImage = []
                    for y in range(Img.size[0]):
                        for x in range(Img.size[1]):
                            newImage.append(255-px[x,y])
                    newImage = np.array(newImage)
                    print(len(newImage))
                    Cc = feedforward([newImage,0],var.b,var.w)
                    print('        Array      =\n',Cc[0:5],'\n',Cc[5:10])
                    print('        File Name  =',name)
                    print('        Largest    =',np.nonzero(Cc == max(Cc))[0][0])
                    print(var.mndata.display(newImage))
                else:
                    print('        get your shit together')
                
                
            
    
        elif choice == '5':
            var.b,var.w = generateRandVars()
            print('    Generated New Random Variables.')

        elif choice == '6':
            print('    Choose Dataset to Use(0 - MNIST / 1 - EMNIST):',end='')
            value = int(input())
            var.imagesSetup(value)
            printParameters()


        print('\nType Desired Function:',end = '')
        choice = input()
            
            






var = variables()
var.setup(
        [784,28,10],    #Network
        0.3,            #eta
        10,             #batchSize
        10,             #repetitions
        1               #Dataset
        )
var.imagesSetup(var.Dataset)

UI()


