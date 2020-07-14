# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:06:41 2019

@author: timme
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scp

WorldData = np.genfromtxt(r"C:\Users\timme\Documents\M.Sc Applied Computation and Numerical Modelling\MA7080 - Mathematical Modelling\ComputationalTask1\GlobalData\GlobalData.csv" ,delimiter=',')
WorldData = WorldData[:,4:]
Year = WorldData[0,:]
GER = WorldData[1,10:] # GER Data starts at Year 10, So 1970 aka reunification
UK = WorldData[2,:]
NOR = WorldData[3,:]
US = WorldData[4,:]
WORLD = WorldData[5,:]
RSA = WorldData[6,:]

def DiscreteLogisticMap(P0,r,K,N):
    # Where N is total number of steps for recursive function
    Pn = np.zeros(N)
    Pn[0] = P0
    for i in np.arange(0,N-1,1):
        Pn[i+1] = r*Pn[i]*(1-(Pn[i]/K))
    return Pn

def rkFVUCalculator(Vector):
    Rand = Vector 
    x = np.polyfit(Rand[:-1],Rand[1:]/Rand[:-1],1)
    m = x[1]
    c = x[0]
    f_x_hat = m*Rand[:-1] + c
    f_x = Rand[1:]/Rand[:-1]
    SS_Err = np.sum((f_x - f_x_hat)**2.)
    f_x_dash = (1/len(f_x))*np.sum(f_x)
    SS_Tot = np.sum((f_x - f_x_dash)**2.)
    r = m
    k = -r/c
    FVU = SS_Err/SS_Tot
    return r,k,FVU,c



LR_RSA = rkFVUCalculator(RSA)
LR_NOR = rkFVUCalculator(NOR)
LR_WORLD = rkFVUCalculator(WORLD)

Sim1_RSA = DiscreteLogisticMap(RSA[0],LR_RSA[0],LR_RSA[1],len(Year))
Sim1_NOR = DiscreteLogisticMap(NOR[0],LR_NOR[0],LR_NOR[1],len(Year))
Sim1_WORLD = DiscreteLogisticMap(WORLD[0],LR_WORLD[0],LR_WORLD[1],len(Year))

plt.plot(Year,Sim1_RSA)
plt.scatter(Year,RSA)
plt.title("Simulation 1 - South Africa")
plt.xlabel("Year")
plt.ylabel("GDP Current US$")

plt.plot(Year,Sim1_NOR)
plt.scatter(Year,NOR)
plt.title("Simulation 1 - Norway")
plt.xlabel("Year")
plt.ylabel("GDP Current US$")

plt.plot(Year,Sim1_WORLD)
plt.scatter(Year,WORLD)
plt.title("Simulation 1 - World")
plt.xlabel("Year")
plt.ylabel("GDP Current US$")
#plt.close("all")
