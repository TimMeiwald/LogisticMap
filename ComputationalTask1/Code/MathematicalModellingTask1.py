# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


def rkFVUCalculator(Vector):
    Rand = Vector 
    x = np.polyfit(Rand[:-1],Rand[1:]/Rand[:-1],1)
    f_x_hat = x[0]*Rand[:-1] + x[1]
    f_x = Rand[1:]/Rand[:-1]
    SS_Err = np.sum((f_x - f_x_hat)**2.)
    f_x_dash = (1/len(f_x))*np.sum(f_x)
    SS_Tot = np.sum((f_x - f_x_dash)**2.)
    r = x[1]
    k = -r/x[0]
    FVU = SS_Err/SS_Tot
    return r,k,FVU,x[0]

def DiscreteLogisticMap(P0,r,K,N):
    # Where N is total number of steps for recursive function
    Pn = np.zeros(N)
    Pn[0] = P0
    for i in np.arange(0,N-1,1):
        Pn[i+1] = r*Pn[i]*(1-Pn[i]/K) 
    return Pn

K = 1.
P0 = np.array([0.5,0.2,0.8])
r= np.array([0.6, 1.7, 2.8, 3.2, 3.6, 3.9])
N = 50


for j in np.arange(0,3,1):
    plt.figure(figsize = (8.0,8.0))
    plt.suptitle("P0 = %f" % P0[j])
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
    for i in np.arange(1,7,1):
        x = "32%d" % i
        plt.subplot(x)
        T = DiscreteLogisticMap(P0[j],r[i-1],K,N)
        plt.plot(np.arange(0,N,1),T)
        plt.title("r = %f" % r[i-1])
        plt.xlabel("Time Steps, N")
        plt.ylabel("Population, P")
        plt.ylim((0,1))
    #plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask1/Figures/DLMP%.2f.png" % j)



for i in np.arange(1,7,1):
    plt.figure()
    plt.figure(figsize = (8.0,8.0))
    plt.suptitle("r = %f" % r[i-1])
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
    for j in np.arange(0,3,1):
        x = "31%d" % j
        plt.subplot(x)
        T = DiscreteLogisticMap(P0[j],r[i-1],K,N)
        plt.plot(np.arange(0,N,1),T)
        plt.title("P0 = %f" % P0[j])
        plt.xlabel("Time Steps, N")
        plt.ylabel("Population, P")
        plt.ylim((0,1))
    #plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask1/Figures/DLMr%.2f.png" % i)

plt.close("all")


WorldData = np.genfromtxt(r"C:\Users\timme\Documents\M.Sc Applied Computation and Numerical Modelling\MA7080 - Mathematical Modelling\ComputationalTask1\GlobalData\GlobalData.csv" ,delimiter=',')
WorldData = WorldData[:,4:]
Year = WorldData[0,:]
GER = WorldData[1,10:] # GER Data starts at Year 10, So 1970 aka reunification
UK = WorldData[2,:]
NOR = WorldData[3,:]
US = WorldData[4,:]
WORLD = WorldData[5,:]
RSA = WorldData[6,:]

dict = {
        "Year" : Year,
        "UK" : UK,
        "NOR": NOR,
        "US" : US,
        "WORLD": WORLD,
        "SAF": RSA,
        "GER" : GER
        }
plt.subplots_adjust(hspace = 0.4)
plt.subplot(321)
plt.plot(Year,NOR)
plt.xlabel("Year")
plt.ylabel("GDP (Current US$)")
plt.title("Norway")
plt.subplot(322)
plt.plot(Year[10:],GER)
plt.xlabel("Year")
plt.title("Germany")
plt.subplot(323)
plt.plot(Year,US)
plt.xlabel("Year")
plt.title("United States of America")
plt.subplot(324)
plt.plot(Year,UK)
plt.xlabel("Year")
plt.title("United Kingdom")
plt.subplot(325)
plt.plot(Year, WORLD)
plt.xlabel("Year")
plt.ylabel("World GDP ")
#plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask1/Figures/WorldGDP.png")
plt.subplot(326)
plt.plot(Year, RSA)
plt.xlabel("Year")
plt.title("South Africa")
plt.close("all")


plt.scatter(GER[:-1],GER[1:]/GER[:-1])
plt.scatter(US[:-1],US[1:]/US[:-1])
plt.scatter(NOR[:-1],NOR[1:]/NOR[:-1])
plt.scatter(UK[:-1],UK[1:]/UK[:-1])
plt.scatter(RSA[:-1],RSA[1:]/RSA[:-1])
plt.scatter(WORLD[:-1],WORLD[1:]/WORLD[:-1])


Rand = RSA
plt.scatter(Rand[:-1],Rand[1:]/Rand[:-1])
plt.title("South African GDP")
plt.xlabel(r"$P_{n}$")
plt.ylabel(r"$\dfrac{P_{n+1}}{P_{n}}$")
x = np.polyfit(Rand[:-1],Rand[1:]/Rand[:-1],1)
plt.plot(Rand[:-1], x[0]*Rand[:-1] + x[1])
f_x_hat = x[0]*Rand[:-1] + x[1]
f_x = Rand[1:]/Rand[:-1]
SS_Err = np.sum((f_x - f_x_hat)**2.)
f_x_dash = (1/len(f_x))*np.sum(f_x)
SS_Tot = np.sum((f_x - f_x_dash)**2.)
r = x[1]
k = -r/x[0]
FRU = SS_Err/SS_Tot
plt.text(3.2*10**11, 1.5,r"r = %f" % r)
plt.text(3.2*10**11, 1.47,r"k = %.2E" % k)
plt.text(3.2*10**11, 1.44,r"FVU = %f" % FRU)
plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask1/Figures/RSAGDPScatter.png")

plt.close("all")


Rand = RSA
plt.scatter(Rand[:-1],Rand[1:]/Rand[:-1])
plt.title("South African GDP")
plt.xlabel(r"$P_{n}$")
plt.ylabel(r"$\dfrac{P_{n+1}}{P_{n}}$")
x1 = rkFVUCalculator(Rand[0:10])
plt.plot(Rand[0:10], x1[3]*Rand[0:10] + x1[0],color = "red")
x2 = rkFVUCalculator(Rand[11:20])
plt.plot(Rand[11:20], x2[3]*Rand[11:20] + x2[0])
x3 = rkFVUCalculator(Rand[21:30])
plt.plot(Rand[21:30], x3[3]*Rand[21:30] + x3[0])
x4 = rkFVUCalculator(Rand[31:40])
plt.plot(Rand[31:40], x4[3]*Rand[31:40] + x4[0])
x5 = rkFVUCalculator(Rand[41:50])
plt.plot(Rand[41:50], x5[3]*Rand[41:50] + x5[0])
x6 = rkFVUCalculator(Rand[50:])
plt.plot(Rand[50:], x6[3]*Rand[50:] + x6[0])
plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask1/Figures/RSAGDPScatter6.png")

plt.close("all")
T = np.column_stack((x1[0:3],x2[0:3],x3[0:3],x4[0:3],x5[0:3],x6[0:3]))


DecadeYear = Year[[5,15,25,35,45,55]]
plt.subplot(311)
plt.suptitle("r, k and Fraction of Variance Unexplained plotted against time interval")
plt.scatter(DecadeYear,T[0])
plt.xlabel("Year")
plt.ylabel("r")
plt.axhline(r,linestyle = "--")
plt.subplot(312)
plt.scatter(DecadeYear,T[1])
plt.xlabel("Year")
plt.ylabel("k")
plt.axhline(k,linestyle = "--")
plt.subplot(313)
plt.scatter(DecadeYear,T[2])
plt.xlabel("Year")
plt.ylabel("FVU")
plt.axhline(FRU,linestyle = "--")
plt.savefig("C:/Users/timme/Documents/M.Sc Applied Computation and Numerical Modelling/MA7080 - Mathematical Modelling/ComputationalTask1/Figures/RgkRSA.png")




print(FRU)

plt.plot(Rand[1:]/Rand[:-1],x[1]*(1- Rand[:-1]/(-x[1]/x[0])))