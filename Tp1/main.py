# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:50:38 2023

@author: pikam
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def genPoisson(l, T, N, distribution):
    
    """
    l : lambda, the packet arrival rate
    T : time window
    N : nb of classes
    distribution : array which contains the distribution for each class
    This function generates packets according to a poisson distribution
    """
    
    packets = []
    
    nbArrivals = np.random.poisson(l)
    for packet in range(nbArrivals) :
       typeEvent = np.random.choice(N, p=distribution) 
       packets.append((typeEvent+1, T))
    return packets
    

def genEpisode():

    distribution = [1/3, 1/12, 1/4, 1/3]  # distribution de chaque événements 
    T = 10  # Intervalle de temps total
    
    # Générer un processus de Poisson multivarié, combien d arrivée pour chaque événements
    numberOfEachEvents = [np.random.poisson(lambda_param * T) for lambda_param in distribution]
    
    # Générer des instants d'arrivée aléatoires pour chaque type d'événement
    arrivalTimesOfEachEvents = []
    for typeOfEvent, numberOfEvent in enumerate(numberOfEachEvents):
        arrivalTimesOfEvent = np.sort(np.random.uniform(0, T, numberOfEvent)) #génére chaque instant d arrivée, entre 0 et T
        arrivalTimesOfEachEvents = arrivalTimesOfEachEvents + ([(typeOfEvent+1, t) for t in arrivalTimesOfEvent])
    
    # Trier tous les instants d'arrivée par ordre chronologique
    #arrivalTimesOfEachEvents.sort(key=lambda x: x[1])
    
    # Afficher les instants d'arrivée
    print("Arrivées :")
    print(arrivalTimesOfEachEvents)
    print(numberOfEachEvents)
    
    return arrivalTimesOfEachEvents


def equation(capacity):
    distributions = [1/3, 1/12, 1/4, 1/3]
    currentCapacity = 0
    alpha = 0
    for distribution in distributions :
        if (currentCapacity + distribution) < capacity :
            currentCapacity = currentCapacity + distribution
            alpha = alpha + 1
        elif currentCapacity < capacity :
            alpha = alpha + ((capacity - currentCapacity) / distribution)
            currentCapacity = currentCapacity + ((capacity - currentCapacity) / distribution)
    return alpha


def equationODE(t, alpha):
    distributions = [1/3, 1/12, 1/4, 1/3]
    currentCapacity = 0
    capacity = 0.34
    alphaInf = math.floor(alpha)
    for j in range(0, alphaInf):
        currentCapacity = currentCapacity + distributions[j]
    currentCapacity = currentCapacity + (alpha - alphaInf)*distributions[(alphaInf+1)-1]
    
    return 0.34 - currentCapacity


def knapsack(packets, capacity):
    currentCapacity = 0
    totalReward = 0
    policy = [0]*len(packets)
    alpha = 0
    rewards = [100, 50, 10, 1]
    distribution = [1/3, 1/12, 1/4, 1/3]
    packets.sort(key=lambda x: x[0])
    print(packets)
    nbEvents = [0,0,0,0]
    nbAccepted = [0,0,0,0]
    for i, (typeEvent, time) in enumerate(packets) :
        nbEvents[typeEvent - 1] = nbEvents[typeEvent - 1] + 1
        if (currentCapacity + distribution[typeEvent - 1]) < capacity :
            currentCapacity = currentCapacity + distribution[typeEvent - 1]
            policy[i] = 1
            nbAccepted[typeEvent - 1] = nbAccepted[typeEvent-1] +1
            totalReward = totalReward + rewards[typeEvent -1]
    print(policy)
    print(totalReward)
    print(nbEvents, nbAccepted)
    return policy      



def robbinsMonro(N, capacity, gamma, e0):
    n = 1
    e = e0 
    #alpha = np.random.uniform(0,N)
    alpha = 1.5
    print(alpha)
    print(math.floor(alpha))
    
    for episode in range(400):
        packets = genPoisson(10**5, 1, 4, [1/3, 1/12, 1/4, 1/3])
        packets.sort(key=lambda x: x[1]) #trier par ordre chrono
        admittedPacquets = 0
        rejectedPacquets = 0
        fraction = alpha - math.floor(alpha)
        
        count = 0
        for numero_classe, temps in packets:
            if numero_classe == (math.floor(alpha)+1):
                count += 1
                
        nbMiddleClass = int((fraction)*count)
        
                
        for (typeEvent, packet) in packets :            
            if typeEvent <= math.floor(alpha) :
                admittedPacquets = admittedPacquets + 1 
            elif typeEvent == (math.floor(alpha)+1) and nbMiddleClass > 0 :
                admittedPacquets = admittedPacquets + 1
                nbMiddleClass = nbMiddleClass - 1
            else :
                rejectedPacquets = rejectedPacquets +1
        Y = admittedPacquets / (admittedPacquets + rejectedPacquets)
        alpha = alpha + e * (capacity-Y)
        alpha = max(0, min(alpha, N))
        n = n +1
        e = e0 / (n**gamma)
        print("alpha : ", alpha)
        
        
        
        
def robbinsMonroEConstant(N, capacity, e):
    n = 1
    #alpha = np.random.uniform(0,N)
    alpha = 2
    print(alpha)
    print(math.floor(alpha))
    for episode in range(200):
        packets = genPoisson(10**5, 1, 4, [1/3, 1/12, 1/4, 1/3])
        packets.sort(key=lambda x: x[1]) #trier par ordre chrono
        admittedPacquets = 0
        rejectedPacquets = 0
        
        fraction = alpha - math.floor(alpha)
        
        count = 0
        for numero_classe, temps in packets:
            if numero_classe == (math.floor(alpha)+1):
                count += 1
                
        nbMiddleClass = int((fraction)*count)
        
        for (typeEvent, packet) in packets :
            print(typeEvent)
            if typeEvent <= math.floor(alpha) :
                admittedPacquets = admittedPacquets + 1 
            elif typeEvent == (math.floor(alpha)+1) and nbMiddleClass > 0 :
                admittedPacquets = admittedPacquets + 1
                nbMiddleClass = nbMiddleClass - 1
            else :
                rejectedPacquets = rejectedPacquets +1
        Y = admittedPacquets / (admittedPacquets + rejectedPacquets)
        alpha = alpha + e * (capacity-Y)
        alpha = max(0, min(alpha, N))
        n = n +1
        print("alpha : ", alpha)
        
        
def polyakAverage(N, capacity, gamma, e0):
    n = 1
    e = e0 
    #alpha = np.random.uniform(0,N)
    alpha = 2
    print(alpha)
    print(math.floor(alpha))
    allY = []
    for episode in range(200):
        packets = genPoisson(10**5, 1, 4, [1/3, 1/12, 1/4, 1/3])
        packets.sort(key=lambda x: x[1]) #trier par ordre chrono
        admittedPacquets = 0
        rejectedPacquets = 0
        
        fraction = alpha - math.floor(alpha)
        
        count = 0
        for numero_classe, temps in packets:
            if numero_classe == (math.floor(alpha)+1):
                count += 1
                
        nbMiddleClass = int((fraction)*count)
        
        for (typeEvent, packet) in packets :
            print(typeEvent)
            if typeEvent <= math.floor(alpha) :
                admittedPacquets = admittedPacquets + 1 
            elif typeEvent == (math.floor(alpha)+1) and nbMiddleClass > 0 :
                admittedPacquets = admittedPacquets + 1
                nbMiddleClass = nbMiddleClass - 1
            else :
                rejectedPacquets = rejectedPacquets +1
        Y = admittedPacquets / (admittedPacquets + rejectedPacquets)
        allY.append(Y)
        averageY = np.mean(allY)
        alpha = alpha + e * (capacity-averageY)
        alpha = max(0, min(alpha, N))
        n = n +1
        e = e0 / (n**gamma)
        print("alpha : ", alpha)
        
        
        
def ode(capacity, nbIterations, gamma, e0):
    listeT = []
    N = 4
    for n in range(1,nbIterations):
        t = 0
        e = e0
        for i in range(1, n):
            e = e0 / (n**gamma)
            t = t + e
        listeT.append(t)
    alpha = np.random.uniform(0,N)
    ode = solve_ivp(equationODE, [listeT[0], listeT[-1]], [alpha], t_eval=listeT)
    
    print(ode.t)
    print(ode.y[0])
        
    
    
    



        
    

"""
T = 10
for second in range(T):
    knapsack(genPoisson(10**5, T, 4, [1/3, 1/12, 1/4, 1/3]), 0.34)
"""
T = 1
#print(equation(0.34))
#knapsack(genPoisson(10**5, T, 4, [1/3, 1/12, 1/4, 1/3]), 0.34*10**5)
#robbinsMonro(4, 0.34, 0.55, 2)

robbinsMonroEConstant(4, 0.34, 0.8)
#polyakAverage(4, 0.34, 0.6, 5)

#ode(0.34, 5000, 0.55, 2)

"""
# Définition de l'ODE
def ode(c, theta):
    return c - theta

# Paramètres de l'ODE
k = 0.1  # Constante k de l'ODE

# Conditions initiales
alpha = 3  # Valeur initiale du taux d'admission
t0 = 0.0  # Temps initial

# Discrétisation du temps
timestep = 0.1
num_steps = 100
times = np.arange(t0, num_steps * timestep, timestep)

# Simulation numérique de l'ODE
alpha_values = [alpha]
for t in times[1:]:
    d_alpha = ode(t, alpha_values[-1]) * timestep
    alpha_new = alpha_values[-1] + d_alpha
    alpha_values.append(alpha_new)

# Traçage des résultats
plt.figure(figsize=(8, 6))
plt.plot(times, alpha_values, label='Trajectoire d\'échantillonnage (Simulation)')
plt.xlabel('Temps')
plt.ylabel('Taux d\'Admission (A)')
plt.title('Simulation de l\'ODE et Trajectoire d\'Échantillonnage')
plt.legend()

# Calcul analytique de la solution de l'ODE
# A_analytique = [A0 * np.exp(-k * t) for t in times]
# plt.plot(times, A_analytique, label='Solution Analytique de l\'ODE', linestyle='--')
# plt.legend()

plt.grid(True)
plt.show()

"""
