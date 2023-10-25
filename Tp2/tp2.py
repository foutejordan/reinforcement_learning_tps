# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 09:50:57 2023

@author: pikam
"""

import numpy as np
import copy


class Gridworld:
    escompte = 0.5

    def __init__(self, k, version):
        assert (version == 1 or version == 2), "erreur dans la version, soit 1 (sans pièges) ou 2 (avec pièges)"
        self.k = k
        self.matriceValue = np.zeros((self.k,self.k))
        self.matriceReward = np.zeros((self.k, self.k))
        self.matriceActions = np.full((self.k, self.k),"N")
        self.start = [0, 0]
        for i in range(0,self.k):
            for j in range(0,self.k):
                self.matriceReward[i][j] = -1
        self.matriceReward[self.k-1][self.k-1] = 2*(self.k-1)
        
        if version == 2:
            
            for j in range(0,8):
                self.matriceReward[3][j] = -2*(self.k-1)
            for j in range(4,12):
                self.matriceReward[7][j] = -2*(self.k-1)

        
    
    def printGridworld(self):
        for ligne in range(self.k-1, -1, -1) :
            for colonne in range(self.k) :
                print(self.matriceValue[ligne, colonne], end="  ")
            print("\n")
            
    def printRewards(self):
        for ligne in range(self.k-1, -1, -1) :
            for colonne in range(self.k) :
                print(self.matriceReward[ligne, colonne], end="  ")
            print("\n")
            
    def printLigneColonne(self):
        for ligne in range(self.k-1, -1, -1) :
            for colonne in range(self.k) :
                print("(",ligne, colonne,")", end="  ")
            print("\n")
            
            
    def recompenseEsperee(self, action, etat): #etat un couple état/action, retourne la récompense espérée
        probas = [1,1,1,1] #probaN(0), probaS(1), probaO(2), probaE(3)
        xN = etat[0]+1
        yN = etat[1] 
        xO = etat[0]
        yO = etat[1]-1
        xS = etat[0]-1
        yS = etat[1]
        xE = etat[0]
        yE = etat[1]+1
        
        #déplacement impossible :
        if xN >= self.k :
            probas[0] = 0
        if yO < 0 :
            probas[2] = 0
        if yE >= self.k :
            probas[3] = 0
        if xS < 0 :
            probas[1] = 0
            
        result = 0
        
        if action == "N":
            if probas[0] == 0 :
                return self.matriceValue[etat[0], etat[1]]
            else :
                result = result + self.matriceValue[xN, yN]*0.8
                
            if probas[2] != 0: #si je peux aller à gauche
                if probas[3] != 0: #si je peux aller à droite
                    return result+ self.matriceValue[xO, yO]*0.1 + self.matriceValue[xE, yE]*0.1
                else :
                    return result + self.matriceValue[xO, yO]*0.2
            elif probas[3] != 0:
                    return result + self.matriceValue[xE, yE]*0.2
            else :
                return result + self.matriceValue[xS, yS]*0.2
                
            
        elif action == "S":
            if probas[1] == 0 :
                return self.matriceValue[etat[0], etat[1]]
            else :
                result = result + self.matriceValue[xS, yS]*0.8
                
            if probas[2] != 0: #si je peux aller à gauche
                if probas[3] != 0: #si je peux aller à droite
                    return result+ self.matriceValue[xO, yO]*0.1 + self.matriceValue[xE, yE]*0.1
                else :
                    return result + self.matriceValue[xO, yO]*0.2
            elif probas[3] != 0:
                    return result + self.matriceValue[xE, yE]*0.2
            else :
                return result + self.matriceValue[xN, yN]*0.2
                
        
        elif action == "O":
            if probas[2] == 0 :
                return self.matriceValue[etat[0], etat[1]]
            else :
                result = result + self.matriceValue[xO, yO]*0.8
                
            if probas[0] != 0: #si je peux aller au nord
                if probas[1] != 0: #si je peux aller au sud
                    return result+ self.matriceValue[xN, yN]*0.1 + self.matriceValue[xS, yS]*0.1
                else :
                    return result + self.matriceValue[xN, yN]*0.2
            elif probas[1] != 0:
                    return result + self.matriceValue[xS, yS]*0.2
            else :
                return result + self.matriceValue[xE, yE]*0.2

        
        elif action == "E":
            if probas[3] == 0 :
                return self.matriceValue[etat[0], etat[1]]
            else :
                result = result + self.matriceValue[xE, yE]*0.8
                
            if probas[0] != 0: #si je peux aller au nord
                if probas[1] != 0: #si je peux aller au sud
                    return result+ self.matriceValue[xN, yN]*0.1 + self.matriceValue[xS, yS]*0.1
                else :
                    return result + self.matriceValue[xN, yN]*0.2
            elif probas[1] != 0:
                    return result + self.matriceValue[xS, yS]*0.2
            else :
                return result + self.matriceValue[xO, yO]*0.2
        
        
    """def allProbas(self):
        actions = ["N","S","O","E"]
        for ligne in range(self.n) :
            for colonne in range(self.m) :
                for action in actions :
                    print(ligne, colonne, action, self.p(action,[ligne,colonne]))"""
                    
                    
                    
    def iteValeur(self): #value iteration
        compteur = 0
        
        converge = False
        while converge == False : #1ere loop
            delta = 0
            self.printGridworld()
            print("\n")
            actions = ["N", "S", "O", "E"]
            for ligne in range(self.k): #2e loop sur les états
                for colonne in range(self.k): #2e loop sur les états
                    temp = self.matriceValue[ligne][colonne] #save old value de v
                    valeurMax = -1000
                    #actionMax = ""
                    for action in actions :
                        valeur = self.recompenseEsperee(action,[ligne,colonne]) #p(s'|s,a)*v(s')
                        if valeur > valeurMax :
                            valeurMax = valeur
                            #actionMax = action
                    compteur = compteur +1
                    self.matriceValue[ligne][colonne] = self.escompte*valeurMax + self.matriceReward[ligne][colonne]
                    #self.matriceActions[ligne][colonne] = actionMax
                    delta = max(delta, abs(temp - self.matriceValue[ligne][colonne]))
            if delta < 0.001:
                converge = True
        #print(compteur)
        #return self.matriceActions
        
        
    def getPolicy(self): #pour avoir les actions de la politique optimale après la value iteration
        
        actions = ["N", "S", "O", "E"]
        for ligne in range(self.k):
            for colonne in range(self.k):
                valeurMax = -1000
                actionMax = ""
                for action in actions :
                    valeur = self.recompenseEsperee(action,[ligne,colonne])
                    if valeur > valeurMax:
                        valeurMax = valeur
                        actionMax = action
                self.matriceActions[ligne][colonne] = actionMax
        return self.matriceActions
            
                
    def iteValeurBis(self, compteur): #partie Evaluation de la Policy Iteration
        converge = False
        while converge == False : #1ere loop
            delta = 0
            self.printGridworld()
            print("\n")
            for ligne in range(self.k): #2e loop sur les états
                for colonne in range(self.k): #2e loop sur les états
                    temp = self.matriceValue[ligne][colonne] #save old value v
                    action = self.matriceActions[ligne][colonne] #action en fonction de la politique
                    valeur = self.recompenseEsperee(action,[ligne,colonne])  #p(s'|s,a)*v(s')
                    compteur = compteur + 1
                    self.matriceValue[ligne][colonne] = self.escompte*valeur + self.matriceReward[ligne][colonne]
                    delta = max(delta, abs(temp - self.matriceValue[ligne][colonne]))
            if delta < 0.001:
                converge = True
        #print(compteur)
        return compteur
                

        
    def getPolicyBis(self): #le reste de la Policy Iteration, la fonction à appeler
        matriceActionPrec = np.full((self.k, self.k),"N")
        converge = False
        compteur = 0
        while converge == False :
            compteur = self.iteValeurBis(compteur)
            matriceActionNew = self.getPolicy()
            converge = True
            for ligne in range(self.k):
                for colonne in range(self.k):
                    if matriceActionPrec[ligne][colonne] != matriceActionNew[ligne][colonne]:
                        converge = False
            for ligne in range(self.k):
                for colonne in range(self.k):
                    matriceActionPrec[ligne][colonne] = matriceActionNew[ligne][colonne]
        return matriceActionPrec
            
    

    
            

gridworld = Gridworld(12, 1)
#gridworld.printGridworld()
gridworld.printRewards()
gridworld.printLigneColonne()
#gridworld.allProbas

#print(gridworld.iteValeur())
gridworld.iteValeur()
print(gridworld.getPolicy())

#print(gridworld.getPolicyBis())
