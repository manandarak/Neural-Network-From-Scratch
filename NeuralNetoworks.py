#!/usr/bin/env python
# coding: utf-8

# In[1]:


inputs =[1,2,3]
weights = [0.2,0.8,-0.5]
bias = 2

output =(inputs[0]*weights[0]+
         inputs[1]*weights[1]+
         inputs[2]*weights[2]+ bias)
output


# In[2]:


inputs =[1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2.0

output =(inputs[0]*weights[0]+
         inputs[1]*weights[1]+
         inputs[2]*weights[2]+ 
         inputs[3]*weights[3]+bias)
output


# In[3]:


inputs = [1,2,3,2.5]
weights1 = [0.2,0.8,-0.5,1]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
    #neuron1
    inputs[0]*weights1[0]+
    inputs[1]*weights1[1]+
    inputs[2]*weights1[2]+
    inputs[3]*weights1[3]+bias1,

    #neuron2
    inputs[0]*weights2[0]+
    inputs[1]*weights2[1]+
    inputs[2]*weights2[2]+
    inputs[3]*weights2[3]+bias2,

    #neuron3
    inputs[0]*weights3[0]+
    inputs[1]*weights3[1]+
    inputs[2]*weights3[2]+
    inputs[3]*weights3[3]+bias3]

print(outputs)


# ## Creating the simple above nn using loops

# In[4]:


inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]


#output layer for each neuron
layers_outputs = []
print(layers_outputs)
#for each neuron

for neuron_weights,neuron_bias in zip(weights,biases):
    #zeroed output of the given neruon
    neuron_output = 0
    print("neruonoutput",neuron_output)
    #for each input and weight of the neuron
    for n_input,weight in zip(inputs,neuron_weights):
         # Multiply this input by associated weight 
        # and add to the neuron’s output variable 
        print("inputs:",inputs)
        print("neuronweights",neuron_weights)
        neuron_output += n_input*weight
        print("neruonoutput",neuron_output)
        # Add bias 
    neuron_output += neuron_bias 
    # Put neuron’s result to the layer’s output list 
    layers_outputs.append(neuron_output) 
 
print(layers_outputs)           


# ## Dot Prodcuts vectors

# In[5]:


a =[1,2,3]
b =[2,3,4]

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(dot_product)


# In[6]:


import numpy as np
inputs = [1.0,2.0,3.0,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2.0

outputs =  np.dot(inputs,weights)+bias
print(outputs)


# In[7]:


import numpy as np
inputs = [1.0, 2.0, 3.0, 2.5] 
weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]] 
biases = [2.0, 3.0, 0.5]
layers_outputs = np.dot(weights,inputs)+biases
print(layers_outputs)


# ## Transposition

# In[8]:


import numpy as np
a = np.array([[1,2,3]])
b = np.array([[2,3,4]]).T  #rows become coloumns and coloumns become rows it flips matrix over its diagonal
print(b)
result = np.dot(a,b)
result


# In[9]:


inputs = [[1.0,2.0,3.0,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
biases = [2.0,3.0,2.5]
outputs = np.dot(inputs,np.array(weights).T)+biases
outputs


# In[10]:


import numpy as np

A = np.array([[1, 2, 3, 2.5],
              [2, 5, -1, 2],
              [-1.5, 2.7, 3.3, -0.8]])

B = np.array([[0.2, 0.5, -0.26],
              [0.8, -0.91, -0.27],
              [-0.5, 0.26, 0.17],
              [1.0, -0.5, 0.87]])

C = np.dot(A, B)
C


# In[11]:


import numpy as np
inputs = [[1,2,3,2.5],
          [2,5,-1,2],
          [-1.5,2.7,3.3,-0.8]]
weights  = [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]

result = np.dot(inputs,np.array(weights).T)+biases
result


# In[12]:


import numpy as np
inputs = [[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]] 

weights = [[0.2, 0.8, -0.5, 1], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]] 
biases = [2, 3, 0.5] 

weights2 = [[0.1, -0.14, 0.5], 
            [-0.5, 0.12, -0.33], 
            [-0.44, 0.73, -0.13]] 

biases2 = [-1, 2, -0.5] 

layer1_outputs = np.dot(inputs,np.array(weights).T)+biases
layers2_outputs = np.dot(layer1_outputs,np.array(weights2).T)+biases2

print(layers2_outputs)


# In[13]:


from  nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt 
X, y = spiral_data(samples=100, classes=3)  # using spiral_data we choose no of class and no of data point we want
plt.scatter(X[:,0], X[:,1]) 
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg') 
plt.show() 


# In[14]:


import numpy as np
import nnfs
nnfs.init()
print(np.random.randn(2,5))


# In[15]:


import numpy as np
import nnfs
nnfs.init()
n_inputs = 2
n_neurons = 4
weights  = 0.1*(np.random.randn(n_inputs,n_neurons))
biases = np.zeros((1,n_neurons))

print(weights)
print(biases)


# In[16]:


# def forward(self,inputs):
#     self.output =  np.dot(inputs,self.weights) + self.biases


# In[17]:


import numpy as np
import nnfs
from nnfs.datasets import spiral_data
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        print(self.weights)
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

x,y = spiral_data(samples=100,classes=3)
X, y = spiral_data(samples =100, classes=3) 
dense1 = Layer_Dense(2, 3)  #2 input features and 3 neurons
dense1.forward(X) 
# print(dense1.output[:5])


# In[18]:


inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []
for i in inputs:
    if i > 0:
        output.append(i)
    else :
        output.append(0)
print(output)


# In[19]:


import numpy as np
inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = np.maximum(0, inputs)
print(output)


# In[20]:


import numpy as np
import nnfs
from nnfs.datasets import spiral_data
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        print(self.weights)
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

class Activation_ReLU:  
    def forward(self, inputs): 
        self.output = np.maximum(0, inputs) 

X, y = spiral_data(samples=100, classes=3) 
dense1 = Layer_Dense(2, 3) 
activation1 = Activation_ReLU()  
dense1.forward(X) 
activation1.forward(dense1.output)
print(activation1.output[:5])


# In[21]:


import numpy as np
X_custom = np.array([[1.0, 2.0],
                     [2.0, 3.0],
                     [-1.0, -2.0]])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        print("Weights:")
        print(self.weights)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# # Creating instances of Layer_Dense and Activation_ReLU
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# # Forward pass with custom input data
dense1.forward(X_custom)
activation1.forward(dense1.output)

 # Print the output of Activation_ReLU for the first 5 examples
print("ReLU Activation Output:")
print(activation1.output[:5])


# In[22]:


import numpy as np
X_custom = np.array([[1.0, 2.0],
                     [2.0, 3.0],
                     [-1.0, -2.0]])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        print("Weights:")
        print(self.weights)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense1.forward(X_custom)
activation1.forward(dense1.output)

print(activation1.output[:5])


# ## Softmax Activation Function

# In[23]:


layer_output = [4.8,1.21,2.385]
E = 2.71828182846
exp_values = []
for output in layer_output:
    exp_values.append(E ** output)
print('exponentiated values:') 
print(exp_values)

norm_base = sum(exp_values)
norm_value = []
for value in exp_values:
    norm_value.append(value/norm_base)
print('Normalized exponentiated values:') 
print(norm_value) 
print('Sum of normalized values:', sum(norm_value)) 


# we can do it same using numpy as below


# In[24]:


import numpy as np
layer_outputs = [4.8,1.21,2.385]
exp_values = np.exp(layer_outputs)
print("exponential values")
print(exp_values)

norm_values = exp_values/np.sum(exp_values)
print("noramlize values")
print(norm_values)


# In[25]:


import numpy as np
layer_outputs = np.array([[4.8,1.21,2.385],
                         [8.9,-1.81,0.2],
                         [1.41,1.051,0.026]])
print('sum without axis')
print(np.sum(layer_outputs))

print('this will be identical to above since default is none')
print(np.sum(layer_outputs,axis = None))


# In[26]:


#axis 0  (ROW) this means to sum row wise along axis 0
print(np.sum(layer_outputs, axis=0))  #4.8+8.9+1.41 #the values from all the other dimensions at this position are summed to form it


# In[27]:


print('So we can sum axis 1, but note the current shape:') 
print(np.sum(layer_outputs, axis=1))


# In[28]:


print('Sum axis 1, but keep the same dimensions as input:') 
print(np.sum(layer_outputs,axis=1,keepdims=True))  #keepdims for dimension as original


# In[29]:


#softmax function
class Activation_Softmax:
    def forward(self,inputs):
        #Getting Unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        #normalizing the probabilities
        probabilites  = exp_values/np.sum(exp_values,axis=1,keepdims = True)
        self.output = probabilites


# In[30]:


import numpy as np
print(np.exp(1))
print(np.exp(10))
print(np.exp(100))
print(np.exp(1000))
print(np.exp(-np.inf),np.exp(0))


# In[31]:


softmax = Activation_Softmax() 
softmax.forward([[1, 2, 3]]) 
print(softmax.output)


# In[32]:


import nnfs
from nnfs.datasets import spiral_data
# Create dataset 
x,y = spiral_data(samples = 100, classes =3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)
# Create ReLU activation (to be used with Dense layer): 
activation1 =  Activation_ReLU()
dense2 = Layer_Dense(3,3)
# Create second Dense layer with 3 input features (as we take output 
# of previous layer here) and 3 output values 
activation2 = Activation_Softmax()
# Create Softmax activation (to be used with Dense layer):
dense1.forward(x)
# Make a forward pass of our training data through this layer
activation1.forward(dense1.output)
# Make a forward pass through activation function 
# it takes the output of first dense layer here 
dense2.forward(activation1.output)
# Make a forward pass through activation function 
# it takes the output of second dense layer here 
activation2.forward(dense2.output)
# Let's see output of the first few samples
print(activation2.output[:5])


# ## Full Code Up to this point

# In[33]:


import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#Dense Layer
class Layer_Dense:

    #Layer initalization
    def __init__(self,n_inputs,n_neurons):
        #Initalize weights and Biases
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    #forward pass
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

#ReLU Activation
class Activation_Relu:
    #forward pass
    def forward(self,inputs):
        #calculate the outputs values from inputs
        self.output = np.maximum(0,inputs)

#Softmax Activation
class Activation_Softmax:
    def forward(self,inputs):
        #get unnormalized probabilites
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims = True))
        #normalize them for each sample
        probabilites = exp_values / np.sum(exp_values,axis=1,keepdims = True)
        self.output = probabilites

#create dataset
X,y = spiral_data(samples=100,classes=3)

#create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)

#create ReLU activation(to be use with Dense Layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output 
# of previous layer here) and 3 output values (output values)
dense2  = Layer_Dense(3,3)

# Create Softmax activation (to be used with Dense layer): 
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer 
dense1.forward(x)

# Make a forward pass through activation function 
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer 
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output) 

# Make a forward pass through activation function 
# it takes the output of second dense layer here
activation2.forward(dense2.output) 

# Let's see output of the first few samples: 
print(activation2.output[:5])


# ## Loss Function 
# ## Cross Entropy

# In[34]:


import math
#An example output from the output layer of the neural network
softmax_output = [0.7,0.1,0.2]
#ground truth
target_output = [1,0,0]

#loss
loss = -(math.log(softmax_output[0])*target_output[0]+
         math.log(softmax_output[1])*target_output[1]+
         math.log(softmax_output[2])*target_output[2])
print(loss)


# In[35]:


softmax_outputs =  [[0.7, 0.1, 0.2], 
                   [0.1, 0.5, 0.4], 
                   [0.02, 0.9, 0.08]] 
class_targets = [0,1,1]


# In[36]:


#spare targets


# In[37]:


import numpy as np
#softmax_outputs 
softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])

#target indicies
class_targets = np.array([0,1,1])

#use the correct class indices to pick the correct cladd probabilites
correct_confidence = softmax_outputs[range(len(softmax_outputs)),class_targets]
print(correct_confidence)

#cal negative log of these probabilities
negative_log_likelihood = -np.log(correct_confidence)

#calculate the mean loss
loss = np.mean(negative_log_likelihood)
print(loss)


# In[38]:


softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])

class_targets = [0,1,1]

for targ_index, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_index]) #conifdence score


# In[39]:


#when targets are in one hot encoded form


# In[40]:


import numpy as np
#softmax outputs
softmax_outputs = [[0.7,0.1,0.2],
                   [0.1,0.5,0.4],
                   [0.02,0.9,0.08]]
#example of one hot encoded targets
one_hot_targets = np.array([[1,0,0],
                            [0,1,0],
                            [0,1,0]])
#example of spare targets 
sparse_targets = np.array([0, 1, 1])


#check if the targets are in the form of one hot encoded
if len(one_hot_targets.shape) == 2:
        # One-hot encoded targets
    correct_confidence = np.sum(softmax_outputs*one_hot_targets, axis=1)
else:
    #sparse targest
    correct_confidence = softmax_outputs[range(len(softmax_outputs)), sparse_targets]

#calculate the negative log
neg_log = -np.log(correct_confidence)

#calculate the average loss
avg_loss = np.mean(neg_log)
print(avg_loss)


# In[41]:


import numpy as np

# Define softmax outputs
softmax_outputs = np.array([[0.7, 0.1, 0.2], 
                            [0.1, 0.5, 0.4], 
                            [0.02, 0.9, 0.08]])

# Define a small value (epsilon) to avoid log(0)
epsilon = 1e-7

# Adjust softmax outputs to avoid zero
adjusted_outputs = softmax_outputs + epsilon

# Example target indices
class_targets = [0, 1, 1]

# Use range(len(softmax_outputs)) to generate [0, 1, 2]
row_indices = range(len(softmax_outputs))

# Select the probabilities for the correct classes
correct_confidence = adjusted_outputs[row_indices, class_targets]

# Calculate negative log with epsilon adjustment
neg_log = -np.log(correct_confidence)

# Calculate average loss
average_loss = np.mean(neg_log)

print(average_loss)


# In[42]:


#Here is how you can implement clipping to avoid the issues with taking the log of 0 or 1:


# In[43]:


import numpy as np

# Example softmax outputs
softmax_outputs = np.array([[0.7, 0.1, 0.2], 
                            [0.1, 0.5, 0.4], 
                            [0.02, 0.9, 0.08]])

# Example target indices
class_targets = [0, 1, 1]

# Clip the softmax outputs
y_pred_clipped = np.clip(softmax_outputs, 1e-7, 1 - 1e-7)
print(y_pred_clipped)

# Select the probabilities for the correct classes
correct_confidence = y_pred_clipped[range(len(softmax_outputs)), class_targets]

# Calculate negative log of the correct confidences
neg_log = -np.log(correct_confidence)

# Calculate the average loss
average_loss = np.mean(neg_log)

print(average_loss)


# # The Categorical Cross-Entropy Loss Class

# In[44]:


#common loss
class Loss:
    #calcualte the data and regularization loss
    def calculate(self,output,y):
        #calculate the sample loss
        sample_losses = self.forward(output,y)
        #calculate the mean loss
        data_loss = np.mean(sample_losses)
        #retunr loss
        return data_loss


# In[45]:


#cross entropy loss class 


# #for the below code refer the code in the cell 53 or just 2-3 cell above this

# In[46]:


import numpy as np

# Define the base loss class
class Loss:
    # Calculate the data and regularization loss
    def calculate(self, output, y):
        # Calculate the sample loss
        sample_losses = self.forward(output, y)
        # Calculate the mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

# Define the categorical cross-entropy loss class
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# # Combining everything up to this point

# In[47]:


import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

#class layer
class Layer_Dense:
    #layer initalization
    def __init__(self,n_inputs,n_neurons):
        #initalize weights and biases 
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        #forward pass
    def forward(self,inputs):
        #calculate output values from inputs,weights and biases 
        self.output = np.dot(inputs,self.weights) + self.biases

#ReLU Activation
class Relu_Activation:
    #forward pass
    def forward(self,inputs):
        #calculate output values from inputs
        self.output = np.maximum(0,inputs)

#softmax_activation
class Activation_Softmax:
    #forward pass
    def forward(self,inputs):
        #get unnormalized probabilitites
        exp_values = np.exp(inputs - np.max(inputs,axis=1,
                                            keepdims= True))
        #normalize them for each samples
        probabilites = exp_values/np.sum(exp_values,axis=1,
                                         keepdims=True)
        self.output = probabilites

#common loss class
class Loss:
    #calculate the data and regularization loss
    #given model output and ground truth values
    def calculate(self,output,y):
        #calculate sample loss
        sample_losses = self.forward(output,y)
        #calcualte mean loss
        data_loss = np.mean(sample_losses)
        #Return loss
        return data_loss

#cross-entropy loss
class Loss_CateogircalCrossentropy(Loss):
    #forward pass
    def forward(self,y_pred,y_true):
        #number of samples in batch
        samples = len(y_pred)
        #clip data to prevent divison by 0
        #clip both sides to not drag mean torwards any values
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        #probabilities for targte values
        #only if categorical labels
        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[
            range(samples),
            y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape)==2:
            correct_confidence == np.sum(
                y_pred_clipped * y_true,
                axis = 1
            )
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
        
X, y = spiral_data(samples=100, classes=3) 
dense1 = Layer_Dense(2, 3) 
activation1 = Relu_Activation() 
dense2 = Layer_Dense(3, 3) 
activation2 = Activation_Softmax() 
loss_function = Loss_CateogircalCrossentropy() 
dense1.forward(X) 
activation1.forward(dense1.output) 
dense2.forward(activation1.output) 
# Perform a forward pass through activation function 
# it takes the output of second dense layer here 
activation2.forward(dense2.output) 
# Let's see output of the first few samples: 
print(activation2.output[:5]) 
# Perform a forward pass through loss function 
# it takes the output of second dense layer here and returns loss 
loss = loss_function.calculate(activation2.output, y) 
print('loss:', loss)
        
    


# ## Accuracy Calculation

# In[48]:


import numpy as np
#probabilites of 3 samples
softmax_outputs = np.array([[0.7,0.2,0.1],
                            [0.5,0.1,0.4],
                            [0.02,0.9,0.08]])
#target (groudn-truth) labels of 3 samples
class_targets = np.array([1,0,1])
#calculate values along second axis (axis of index 1)
predicitions = np.argmax(softmax_outputs,axis=1)
#if targets are one hot encoded convert them
if len(class_targets.shape)==2:
    class_targets=np.argmax(class_targets,axis=1)

#True evaluates to 1,False to 0
accuracy = np.mean(predicitions == class_targets)
print('acc:',accuracy)


# In[49]:


import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()
X,y = vertical_data(samples=100,classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg') 
plt.show()


# In[50]:


X, y = vertical_data(samples=100, classes=3) 
# Create model 
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs 
activation1 = Activation_ReLU() 
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs 
activation2 = Activation_Softmax() 
# Create loss function 
loss_function = Loss_CategoricalCrossentropy()


# In[51]:


#helper varaible
lowest_loss = 9999999 #some random value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


# In[52]:


import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

# Generate dataset
X, y = vertical_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

# Define the necessary classes (Assuming these are already defined in your previous code)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss_CategoricalCrossentropy:
    def calculate(self, output, y):
        samples = len(output)
        y_pred_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        if len(y.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

# Create model
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999  # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(1000):
    # Generate a new set of weights for iteration
    #random weights and biases are generated for both layer
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)

    

    # Perform a forward pass of the training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss


# In[53]:


#Better Approach for optimizing the nerual network loss:
# How it works:

# Small Adjustments: We add small random values to the current best weights and biases. This is like making a small step in a random direction.
# Evaluate: We calculate the loss with the new weights and biases.
# Decision:
# If the loss decreases, it means the new weights and biases are better. We keep these new values as the best ones.
# If the loss increases, it means the step was in the wrong direction. We revert back to the previous best weights and biases.

# dense1.weights += 0.05 * np.random.randn(2,3)
# dense1.biases += 0.05 * np.random.randn(1,3)
# dense2.weights += 0.05 * np.random.rand(3,3)
# dense2.baises += 0.05 * np.random.rand(1,3)


# # changing our ending if statement to :- 
# # If loss is smaller - print and save weights and biases aside 
# if loss < lower_loss:
#      print('New set of weights found, iteration:', iteration, 
#               'loss:', loss, 'acc:', accuracy)
#     best_dense1_weights = dense1.weights.copy()
#     best_dense1_biases = dense1.biases.copy()
#     best_dense2_weights = dense2.weights.copy()
#     best_dense2_baises  = dense2.biases.copy()
#     lowest_loss = loss
# #rever weights and biases
# else :
#      dense1.weights = best_dense1_weights.copy
#      dense1.biases = best_dense1_biases.copy
#      dense2.weights = best_dense2_weights.copy
#      dense2.biases = best_dense2_biases.copy


# In[54]:


import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

# Generate dataset
X, y = vertical_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

# Define the necessary classes (Assuming these are already defined in your previous code)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss_CategoricalCrossentropy:
    def calculate(self, output, y):
        samples = len(output)
        y_pred_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        if len(y.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

# Create model
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999  # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(1000):
    # Generate a new set of weights for iteration
    #random weights and biases are generated for both layer
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.rand(3,3)
    dense2.biases += 0.05 * np.random.rand(1,3)

    

    # Perform a forward pass of the training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss: 
        print('New set of weights found, iteration:', iteration, 
              'loss:', loss, 'acc:', accuracy) 
        best_dense1_weights = dense1.weights.copy() 
        best_dense1_biases = dense1.biases.copy() 
        best_dense2_weights = dense2.weights.copy() 
        best_dense2_biases = dense2.biases.copy() 
        lowest_loss = loss 
    # Revert weights and biases 
    else: 
        dense1.weights = best_dense1_weights.copy() 
        dense1.biases = best_dense1_biases.copy() 
        dense2.weights = best_dense2_weights.copy() 
        dense2.biases = best_dense2_biases.copy() 


# ## Chapter 7 Derivatives

# In[55]:


def f(x):
    return 2*x


# In[56]:


import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return 2*x
x = np.array(range(5))
y = f(x)
print(x)
print(y)

plt.plot(x,y)
plt.show()


# In[57]:


print((y[1]-y[0]) / (x[1]-x[0])) 


# In[58]:


import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return 2*x**2  
x = np.array(range(5))
y = f(x)
print(x)
print(y)

plt.plot(x,y)
plt.show()


# In[59]:


#first pairs 
print((y[1]-y[0]) / (x[1]-x[0])) 


# In[60]:


print((y[3]-y[2]) / (x[3]-x[2])) 


# ## Calculating the derivative for the non linearity

# In[61]:


p2_delta = 0.0001
x1 = 1
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

approx_derivative = (y2-y1)/(x2-x1)
print(approx_derivative)


# In[62]:


p2_delta = 0.0001 
x1 = 1 
x2 = x1 + p2_delta  # add delta 
y1 = f(x1)  # result at the derivation point 
y2 = f(x2)  # result at the other, close point 
approximate_derivative = (y2-y1)/(x2-x1) 
print(approximate_derivative)


# In[63]:


import matplotlib.pyplot as plt 
import numpy as np
def f(x):
    return 2*x**2
# np.arange(start, stop, step) to give us smoother line
x =  np.arange(0,5,0.001)
y = f(x)
plt.plot(x,y)
plt.show()
print(x)


# In[64]:


b = y2 - approximate_derivative*x2


# In[65]:


import matplotlib.pyplot as plt 
import numpy as np

def f(x):
    return 2*x**2 

#np.arrange(start,stop,step) to give us smoother line
x = np.arange(0,5,0.001)
y = f(x)

plt.plot(x,y)

#the point and the close enough point 
p2_delta = 0.0001
x1 = 2
x2 = x1+p2_delta

y1 = f(x1)
y2 = f(x2)

print((x1,y1),(x2,y2))

# Derivative approximation and y-intercept for the tangent line 
approximate_derivative = (y2-y1)/(x2-x1) 
b = y2 - approximate_derivative*x2

# We put the tangent line calculation into a function so we can call 
# it multiple times for different values of x 
# approximate_derivative and b are constant for given function 
# thus calculated once above this function

def tangent_line(x): 
    return approximate_derivative*x + b 

# plotting the tangent line 
# +/- 0.9 to draw the tangent line on our graph 
# then we calculate the y for given x using the tangent line function 
# Matplotlib will draw a line for us through these points 

to_plot = [x1-0.9, x1, x1+0.9] 
plt.plot(to_plot, [tangent_line(i) for i in to_plot]) 
print('Approximate derivative for f(x)', 
      f'where x = {x1} is {approximate_derivative}') 
plt.show() 
print(y2)
print(x2)


# In[66]:


print(x)


# In[1]:


import matplotlib.pyplot as plt 
import numpy as np 
def  f(x): 
return 2*x**2 
# np.arange(start, stop, step) to give us a smoother curve 
x = np.array(np.arange(0,5,0.001)) 
y = f(x) 
plt.plot(x, y) 
colors = ['k','g','r','b','c'] 
def ​ approximate_tangent_line(x​, approximate_derivative​): 
return (approximate_derivative*x) + b 
Chapter 7 - Derivatives - Neural Networks from Scratch in Python 
 
19 
for i in range(5): 
    p2_delta = 0.0001 
    x1 = i 
    x2 = x1+p2_delta 
 
    y1 = f(x1) 
    y2 = f(x2) 
 
    print((x1, y1), (x2, y2)) 
    approximate_derivative = (y2-y1)/(x2-x1) 
    b = y2-(approximate_derivative*x2) 
 
    to_plot = [x1-0.9, x1, x1+0.9] 
 
    plt.scatter(x1, y1, c=colors[i]) 
    plt.plot([point for point in to_plot], 
             [approximate_tangent_line(point, approximate_derivative) 
                 for point in to_plot], 
             c=colors[i]) 
 
    print('Approximate derivative for f(x)', 
          f'where x = {x1} is {approximate_derivative}') 
 
plt.show()


# In[ ]:




