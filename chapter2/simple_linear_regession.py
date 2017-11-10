import os 
import math
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "datasets/"

column_names = ["x", "y"]
data = pd.read_csv("datasets/data.csv", names=column_names)
print(data)

train_x = list()
train_y = list()
for i in range(99) :
    train_x.append(data["x"][i])
    train_y.append(data["y"][i])
    
def loss_function(m, b, train_x, train_y):
    n = len(train_x)
    totalError = 0
    for i in range(n) : 
        x = train_x[i]
        y = train_y[i]
        totalError += (y - (m*x + b))**2 # Sum Square Error (SSE)
    return totalError/float(n)
        
    
def gradient_descent(m_starting, b_starting, train_x, train_y, a, n) :
    m_gradient = 0
    b_gradient = 0
    new_m = m_starting
    new_b = b_starting
    for i in range(n) :        
        x = train_x[i]
        y = train_y[i]
        m_gradient += -(2/n)*x*(y - (m_starting*x + b_starting))
        b_gradient += -(2/n)*(y - (m_starting*x + b_starting))
    new_m += -(a*m_gradient)
    new_b += -(a*b_gradient)
    return new_m, new_b
    
    
# predict y = m*x + b
def train():
    learning_rate = 0.0001
    num_iterations = 1000
    m = 0 # initial m
    b = 0 # initial b
    n = len(train_x) # get size of data
    # training 
    for i in range(num_iterations) :            
        m, b = gradient_descent(m, b, train_x, train_y, learning_rate, n)        
        if (i % 100 == 0): 
            # visualize 
            x0 = np.linspace(1, 1000,120)
            y0 = m*x0 + b
            plt.axis([0,120, 0, 120])
            plt.plot(x0, y0,'-')
            plt.plot(train_x, train_y, 'o')
            plt.show()            
            print("Error = " + str(loss_function(m, b, train_x, train_y)))
    print("y = " + str(m) +  "*x + " + str(b))
    print("Error = " + str(loss_function(m, b, train_x, train_y)))
    return m,b
train()
