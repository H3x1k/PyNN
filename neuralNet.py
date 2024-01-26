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

# 1 is good
# 0 is bad

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
  
def sigmoid(x):
  return np.exp(x)/(np.exp(x)+1)
    
class Layer:
  weights = []
  biases = []
  def __init__(self, node_size, input_size, activation_function = None):
    self.node_size = node_size
    self.output_size = node_size
    self.input_size = input_size
    self.activation_function = activation_function
    for i in range(input_size*2):
      self.weights.append(5*random()+1)
    for i in range(input_size):
      self.biases.append(0)

  def computeOutput(self, input):
    if len(input) != self.input_size:
      raise ValueError("wrong amount of inputs")
    out = []
    for i in range(self.output_size):
      x = 0
      for j in range(len(input)):
        x += input[j] * self.weights[self.input_size * i + j]
      if self.activation_function == None:
        out.append(x)
      else:
        out.append(self.activation_function(x))
    return out

class NeuralNetwork:
  layers = []
  def __init__(self, input_size, output_size, layer_size, node_size, activation_function = ReLU):
    self.input_size = input_size
    self.output_size = output_size
    for i in range(layer_size):
      if i == 1:
        self.layers.append(Layer(node_size, input_size, activation_function))
      elif i == layer_size - 1:
        self.layers.append(Layer(output_size, node_size))
      else:
        self.layers.append(Layer(node_size, node_size, activation_function))

  def computeOutput(self, input):
    if len(input) != self.input_size:
      raise ValueError("wrong amount of inputs")
    values = input
    for i in self.layers:
      values = i.computeOutput(values)
    return values
  
def createData(function, D, R, amount):
  xData = []
  yData = []
  zData = []
  for i in range(amount):
    x = (D * random()) - (D / 2)
    y = (R * random()) - (R / 2)
    xData.append(x)
    yData.append(y)
    if y > function(x):
      zData.append(0)
  return [xData, yData, zData]

def randomFunction(x):
  a = 2.075821551294103
  b = -1.8407634918907476
  c = -2.8569862359719056
  d = 3.682377752576315
  return a*x**3 + b*x**2 + c*x + d

dataAmount = 100

NN = NeuralNetwork(2, 2, 1, 2)
data = createData(randomFunction, 20, 20, dataAmount)

for i in range(dataAmount):
  print(NN.computeOutput([data[0][i],data[1][i]]))