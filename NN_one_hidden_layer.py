import matplotlib.pyplot as plt
import numpy as np
import time

class Artificial_Neural_Network(object):
      def __init__(self):
        
        self.numberOfInputs = 1
        self.numberOfoutputs = 1
        self.numberOfNuerons = 8
        self.learning_rate =0.3
        
        #momentum
        self.beta =0.88
        self.velocity_1 = 0
        self.velocity_2 = 0
        self.velocity_b1 = 0
        self.velocity_b2 = 0
        self.hidden_outputs = []
       
        #we can use xavier initialization
        """
        self.weight_1 = (np.random.rand(self.numberOfInputs, self.numberOfNuerons) * np.sqrt(2/self.numberOfNuerons))
        self.weight_2 =(np.random.rand(self.numberOfNuerons, self.numberOfoutputs) * np.sqrt(2/self.numberOfNuerons))
        self.bias_1 = np.random.rand(1,self.numberOfNuerons) *np.sqrt(2/self.numberOfNuerons)
        self.bias_2  = np.random.rand(1,1) * np.sqrt(2/self.numberOfNuerons)
        """
        
        self.weight_1 = (((np.random.rand(self.numberOfInputs, self.numberOfNuerons))*2)-1)
        self.weight_2 = (((np.random.rand(self.numberOfNuerons, self.numberOfoutputs))*2)-1)
        self.bias_1 = (((np.random.rand(1,self.numberOfNuerons))*2 )-1)
        self.bias_2  = (((np.random.rand(1,1))*2)-1)
    
           
      def relu(self,a):             
            return  np.maximum(a,0)    
         
      def reluDerivative(self,da):       
            da[da<=0] = 0
            da[da>0] = 1
            return da  
    
      def sigmoid(self, a):    
          return 1/(1+np.exp(-a))
    
      def sigmoidDerivative(self, da):   
          return da * (1 - da)
        
      def feedforward(self, x):            
            
            self.dot_production = np.dot(x, self.weight_1) + self.bias_1                       
            self.layer_1 = self.sigmoid(self.dot_production) 
            self.hidden_outputs = self.layer_1  
            self.output = np.dot(self.layer_1, self.weight_2) + self.bias_2 
            
            return self.output
            
      def backpropogate(self, x, y, output):
                    
            self.error = output - y 
            self.error = 2*self.error
                       
            self.layer_1_der = (self.error.dot(self.weight_2.T))*self.sigmoidDerivative(self.layer_1) 
            der_weight_1 = ((x.T.dot(self.layer_1_der))*self.learning_rate) /x.shape[0]
            der_weight_2 = ((self.layer_1.T.dot(self.error))*self.learning_rate) /x.shape[0]
            self.velocity_1 = self.beta *self.velocity_1 + der_weight_1
            self.velocity_2 = self.beta *self.velocity_2 + der_weight_2
            
            der_b1 =  np.sum(self.layer_1_der*self.learning_rate,axis=0,keepdims=True) / x.shape[0] 
            
            der_b2 = np.sum((self.error)*self.learning_rate)/ x.shape[0]
            self.velocity_b1  = self.beta *self.velocity_b1 + der_b1
            self.velocity_b2 = self.beta *self.velocity_b2 + der_b2
                    
            self.weight_1 -= self.velocity_1
            self.weight_2 -= self.velocity_2
            self.bias_1 -=  self.velocity_b1
            self.bias_2 -= self.velocity_b2
            
     
      def denormalize(self,myData,std_data,mean_data):
            myData = (myData * std_data) + mean_data
            return myData
        
      def normalize(self,myData):
            myData  = (myData - np.mean(myData))/np.std(myData)
            return myData
        
      def train (self, x, y):           
            output = self.feedforward(x)
            self.backpropogate(x, y, output)
        
      def predict(self,x):
            return self.feedforward(x)  
        
      def loss(self,x,y):
            return  np.mean(np.square(y - self.predict(x)))
        
      def loss2(self,test,y):
            return  np.mean(np.square(y - test))

my_Nueral_Net = Artificial_Neural_Network()

    
loadedData = np.loadtxt("train1.txt")
x0 = [row[0] for row in loadedData]

y0 = [row[1] for row in loadedData]
x0 = np.array(x0)
y0 = np.array(y0)
x0 = x0.reshape(x0.shape[0],1)
y0 = y0.reshape(y0.shape[0],1)

stdX = np.std(x0)
meanX = np.mean(x0)

stdY= np.std(y0)
meanY = np.mean(y0)

# scale units
x_train  = my_Nueral_Net.normalize(x0)
y_train = my_Nueral_Net.normalize(y0)

#number of epoches
epoch =10000
learning_method = "batch"
#batch learning
if (learning_method =="batch"):    
    start_time = time.time()
    
    while epoch > 0:          
      if epoch % 10 ==0:
          print ("Loss Train: \n" + str(my_Nueral_Net.loss(x_train,y_train)))
          print ("\n")
      
      my_Nueral_Net.train(x_train, y_train)
      epoch = epoch - 1
    print("--- %s seconds ---" % (time.time() - start_time))
#stochastic GD
else:
     start_time = time.time()
     while epoch > 0:  
         i = 0        
         for instance in x_train:                          
             print ("\n")
             my_Nueral_Net.train(instance, y_train[i])
             i = i + 1
         epoch = epoch - 1    
     print("--- %s seconds ---" % (time.time() - start_time))
     print ("Loss Train: \n" + str(my_Nueral_Net.loss(x_train,y_train)))


my_predict = my_Nueral_Net.feedforward(x_train)


x_train = my_Nueral_Net.denormalize(x_train,stdX,meanX)
y_train = my_Nueral_Net.denormalize(y_train,stdY,meanY)

my_predict = my_Nueral_Net.denormalize(my_predict,stdY,meanY)

x2 = x_train

x2, my_predict = zip(*sorted(zip(x2, my_predict)))
plt.plot(x2,my_predict)

x_train, y_train = zip(*sorted(zip(x_train, y_train)))

plt.scatter(x_train,y_train,label = "Training Data")

plt.plot(x_train,my_predict,label = "Predicted Data")
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/train1.png')
plt.show()

#we can use this part to plot the hidden layer outputs
xx1 = x0
xx2 = x0
xx3 = x0
xx4 = x0
xx5 = x0
xx6 = x0
xx7 = x0
xx8 = x0

vv = [row[0] for row in my_Nueral_Net.hidden_outputs]
xx1, vv = zip(*sorted(zip(xx1, vv)))
plt.plot(xx1,vv)
vv = [row[1] for row in my_Nueral_Net.hidden_outputs]
xx2, vv = zip(*sorted(zip(xx2, vv)))
plt.plot(xx2,vv)

vv = [row[2] for row in my_Nueral_Net.hidden_outputs]
xx3, vv = zip(*sorted(zip(xx3, vv)))
plt.plot(xx3,vv)

vv = [row[3] for row in my_Nueral_Net.hidden_outputs]
xx4, vv = zip(*sorted(zip(xx4, vv)))
plt.plot(xx4,vv)

txt = "Hidden units output for first training set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/sigmoid_layer_train1.png')
plt.show()


"""
vv = [row[4] for row in my_Nueral_Net.hidden_outputs]
xx5, vv = zip(*sorted(zip(xx5, vv)))
plt.plot(xx5,vv)

vv = [row[5] for row in my_Nueral_Net.hidden_outputs]
xx6, vv = zip(*sorted(zip(xx6, vv)))
plt.plot(xx6,vv)

vv = [row[6] for row in my_Nueral_Net.hidden_outputs]
xx7, vv = zip(*sorted(zip(xx7, vv)))
plt.plot(xx7,vv)
vv = [row[7] for row in my_Nueral_Net.hidden_outputs]
xx8, vv = zip(*sorted(zip(xx8, vv)))
plt.plot(xx8,vv)
txt = "Hidden units outputfor second training set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/sigmoid_layer_train1.png')
plt.show()
"""
"""

"""



loadedData2 = np.loadtxt("test1.txt")
x1 = [row[0] for row in loadedData2]
y1 = [row[1] for row in loadedData2]
x1 = np.array(x1)
y1 = np.array(y1)
x1 = x1.reshape(x1.shape[0],1)
y1 = y1.reshape(y1.shape[0],1)
z = x1+y1

stdX1 = np.std(x1)
meanX1 = np.mean(x1)

stdY1= np.std(y1)
meanY1 = np.mean(y1)

"""
VERY IMPORTANT :
    What I changed is in this part
    I was wrongly normalizing my test set based on its own mean
    and variance, which caused shifting the plot to write or left
    (because of diffirence in mean and varinance.)
    what I changed was normalizing it based on train set's mean and variance.
"""
#x_test  = my_Nueral_Net.normalize(x0)
#y_test = my_Nueral_Net.normalize(y0)
x1_normalized=(x1 - np.mean(x0))/np.std(x0)
test = my_Nueral_Net.feedforward(x1_normalized)


x1_normalized=(x1 - np.mean(x0))/np.std(x0)
y1_normalized=(y1 - np.mean(y0))/np.std(y0)
test_preds=my_Nueral_Net.feedforward(x1_normalized)
test_loss_value=np.mean(np.power((y1_normalized - test_preds),2))
print ("Loss Test: \n" + str(test_loss_value))

x_test  = my_Nueral_Net.denormalize(x1_normalized,stdX,meanX)
y_test = my_Nueral_Net.denormalize(y1_normalized,stdY,meanY)
test = my_Nueral_Net.denormalize(test,stdY,meanY)

x2 = x_test
#print(x_test)
x2, test = zip(*sorted(zip(x2, test)))

plt.plot(x2,test)


x_test, y_test = zip(*sorted(zip(x_test, y_test)))

plt.scatter(x_test,y_test, c='b' ,label = "Test Data")
plt.plot(x_test,test,label = "Predicted")
txt = "Test sample"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/test1.png')
plt.show()

