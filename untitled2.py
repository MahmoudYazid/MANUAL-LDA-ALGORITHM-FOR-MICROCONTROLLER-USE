
#LDA Algorithms.

import numpy as np


class var():
  x=[1,1.5,1.9,2.5,2,1,2]
  y=[1,1,1,2,2,1,2]
  ClassesCount=len(np.unique(var.y))
  

def MeanOfClass1():return np.mean([var.x[i] for i in range(0,len(var.y)) if var.y[i]==1])
  
 
def MeanOfClass2():return np.mean([var.x[i] for i in range(0,len(var.y)) if var.y[i]==2])
  

def pClass1(): return len([var.x[i] for i in range(0,len(var.y)) if var.y[i]==1])/len(var.y)

def pClass2(): return len([var.x[i] for i in range(0,len(var.y)) if var.y[i]==2])/len(var.y)

def CountClass1():return len([var.x[i] for i in range(0,len(var.y)) if var.y[i]==1])

def CountClass2():return len([var.x[i] for i in range(0,len(var.y)) if var.y[i]==2])

def SquaredDifferancesClass1():return np.sum([(var.x[x]-MeanOfClass1())**2  for x in range(0,len(var.x)) if var.y[x]==1])

def SquaredDifferancesClass2():return np.sum([(var.x[x]-MeanOfClass2())**2  for x in range(0,len(var.x))if var.y[x]==2])

def VarianceClass1(): return (1/CountClass1()-var.ClassesCount) * (SquaredDifferancesClass1()+SquaredDifferancesClass2())

def VarianceClass2(): return (1/CountClass2()-var.ClassesCount) * (SquaredDifferancesClass1()+SquaredDifferancesClass2())

def discriminantForClass1(x_predict):return x_predict *(MeanOfClass1()/VarianceClass1())-(MeanOfClass1()**2/2*VarianceClass1())+np.log(pClass1())

def discriminantForClass2(x_predict):return x_predict *(MeanOfClass2()/VarianceClass2())-(MeanOfClass2()**2/2*VarianceClass2())+np.log(pClass2())

def predict_array(x_predict) : return [discriminantForClass1(x_predict),discriminantForClass2(x_predict)]

def predict(x_predict):return [  2 if predict_array(x_predict)[0]>predict_array(x_predict)[1] else 1 ] 

if __name__=="__main__":

  x=predict(.4)
  print(x)
  x=predict(7)
  print(x)
