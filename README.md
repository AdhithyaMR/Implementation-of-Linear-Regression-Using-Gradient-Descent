# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Adhithya M R
RegisterNumber:  212222240002
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header =None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def computeCost(X,y,theta):

  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  j=1/(2*m)* np.sum(square_err)
  return j

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range (num_iters):
    predictions=X.dot(theta)
    error = np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history  

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1" )

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Grading Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict (x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population =70,000,we predict a profit a profit of $"+str(round(predict2,0)))

```

## Output:
### COMPUTE COST VALUE
![270317953-ad1784a6-3514-450b-a9c5-13bb3a0c25ce](https://github.com/AdhithyaMR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118834761/18b1361b-ccfe-4f4c-961c-ec32ccadd3cd)
### H(X) VALUE
![270318174-7db17e1e-b899-4e94-b5d1-acbdc060dd2a](https://github.com/AdhithyaMR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118834761/dbfb24f2-71fa-42f7-ab54-c002abf63bef)
### COST FUNCTION USING GRADIENT DESCENT GRAPH
![270318303-d4b0d82c-474e-4a8b-9ae9-7be5882c9bbc](https://github.com/AdhithyaMR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118834761/b1d60344-f5bc-42d3-95d8-92a27284a016)
### PROFIT PREDICTION GRAPH

![270318452-aaabb494-5453-4f8a-a40e-875076aea92e](https://github.com/AdhithyaMR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118834761/a9d130e3-91de-4cca-89ef-a72b617237f2)

### PROFIT FOR THE POPULATION 35,000

![270318582-77a040fa-3800-4368-b73c-f02b5e27d8eb](https://github.com/AdhithyaMR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118834761/96844b54-e8ae-4134-bef5-d2a5275c63be)

### PROFIT FOR THE POPULATION 70,000

![270318727-32d57d96-0d39-4e70-984d-8cd7ff5589bb](https://github.com/AdhithyaMR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118834761/d7716f91-3e8c-40d9-9278-699da534d6b1)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
