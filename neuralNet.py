import matplotlib.pyplot as plt
import numpy as np
from random import random
#import TKinter

#def f(x):
  #return np.sin(x)
#x = np.linspace(0,10,1)
#y = f(x)
#plt.plot(x,y, color='black')
#plt.show()

# 1, 0 is good
# 0, 1 is bad

def identity(x):
  return x

def binary(x):
  if x < 0:
    return 0
  else:
    return 1

def ReLU(x):
  if x <= 0:
    return 0
  else:
    return x
  
def sigmoid(x, derivative=False):
  if derivative:
    return sigmoid(x)*(1-sigmoid(x))
  else:
    return 1/(np.exp(-x)+1)
    
class Layer:
  weights = [] # 2D Array weight[input][output]
  biases = []
  inputs = []
  weightActivations = []
  def __init__(self, node_size, input_size, activation_function = None):
    self.node_size = node_size
    self.output_size = node_size
    self.input_size = input_size
    self.activation_function = activation_function
    for j in range(input_size):
      nodeWeights = []
      for k in range(node_size):
        nodeWeights.append(5*random()+1)
      self.weights.append(nodeWeights)
    for i in range(input_size):
      self.biases.append(0)

  def computeOutput(self, input):
    if len(input) != self.input_size:
      raise ValueError("wrong amount of inputs")
    out = []
    for i in range(self.output_size):
      x = self.biases[i]
      for j in range(len(input)):
        x += input[j] * self.weights[j][i]
      out.append(self.activation_function(x))
    return out
  
  def activationFunction(self, input):
    if self.activation_function == None:
      return input
    else:
      return self.activation_function(input)

class NeuralNetwork:
  layers = []
  def __init__(self, input_size, output_size, Hlayer_size, node_size, activation_function = None):
    self.input_size = input_size
    self.output_size = output_size
    for i in range(Hlayer_size + 1):
      if i == 0: # First Hidden Layer
        self.layers.append(Layer(node_size, input_size, activation_function))
      elif i == Hlayer_size: # Output Layer
        self.layers.append(Layer(output_size, node_size, activation_function))
      else: # Hidden Layer
        self.layers.append(Layer(node_size, node_size, activation_function))

  def computeOutput(self, input):
    if len(input) != self.input_size:
      raise ValueError("wrong amount of inputs")
    for i in self.layers:
      input = i.computeOutput(input)
    return input
  
  def nodeCost(self, output, expected):
    error = output - expected
    return error * error
  
  #def Cost(self, dataPoint):

  
  def calculateDescent(self, data):

    weightsChanges = []
    biasChanges = []

    for l in self.layers:
      layerWeights = []
      for w in l.weights:
        layerWeights.append(0)
      weightsChanges.append(layerWeights)

    for l in self.layers:
      layerBiases = []
      for b in l.biases:
        layerWeights.append(0)
      biasChanges.append(layerBiases)

    for i in range(len(data[0])):
      layerWeights = []
      for li in range(len(weightsChanges)):
        for l in range(li + 1):
          for i in range(len(l.weights)):
            layerWeights[i] = 

  
  def train(self, data):
    self.calculateDescent(data)
  
def createData(function, D, R, amount):
  Data = []
  for i in range(amount):
    dataPoint = []
    x = (D * random())
    y = (R * random())
    dataPoint.append(x)
    dataPoint.append(y)
    if y > function(x):
      dataPoint.append(0)
    else:
      dataPoint.append(1)
    Data.append(dataPoint)
  return Data

def randomFunction(x):
  b = -2.8407634918907476
  c = 2.8569862359719056
  d = 2
  return b*x**2 + c*x + d

dataAmount = 1000
dataDomain = 3
dataRange = 3

NN = NeuralNetwork(2, 2, 1, 2, sigmoid)
data = createData(randomFunction, dataDomain, dataRange, dataAmount)

correctAmount = 0
for i in range(dataAmount):
  estimate = NN.computeOutput([data[i][0],data[i][1]])
  #print(estimate)
  if estimate[0] > estimate[1]:
    if data[i][0] == 1:
      correctAmount += 1
  elif estimate[1] > estimate[0]:
    if data[i][1] == 0:
      correctAmount += 1

print((correctAmount/dataAmount)*100, "% correct")