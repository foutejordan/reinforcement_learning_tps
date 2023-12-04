import numpy as np

# Define the parameters of the MDP
k = 12
version = 1 #version 1 ou 2

# Initial state
start_state = (11, 0)

# Reward matrix
rewards = np.zeros((k, k))
for i in range(0, k):
    for j in range(0, k):
        rewards[i, j] = -1
rewards[0, k - 1] = 2 * (k - 1)

if version == 2 :
    for j in range(0, 8):
        rewards[8][j] = -2 * (k - 1)
    for j in range(4, 12):
        rewards[4][j] = -2 * (k - 1)

gamma = 0.9


#print(rewards)

# Stopping condition for value iteration
epsilon = 0.001

# Initialize the values of all states to 0
values = np.zeros((k, k))

def valueIteration():
    # Apply the value iteration algorithm
    while True:
        delta = 0
        for i in range(0, k):
            for j in range(0, k):
                state = (i, j)
                max_value = float('-inf')
                for action in ["N", "S", "O", "E"]:
                    next_state = state
                    if next_state in [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (4, 4), (4, 5),
                                      (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11)]:
                        continue
                    if action == "N":
                        next_state = (i - 1, j)
                    elif action == "S":
                        next_state = (i + 1, j)
                    elif action == "O":
                        next_state = (i, j - 1)
                    elif action == "E":
                        next_state = (i, j + 1)
    
                    if 0 <= next_state[0] < k and 0 <= next_state[1] < k:
                        action_value = rewards[next_state[0], next_state[1]] + gamma * values[next_state[0], next_state[1]]
                        if action_value > max_value:
                            max_value = action_value
    
                delta = max(delta, abs(values[i, j] - max_value))
                values[i, j] = max_value
    
        if delta < epsilon:
            break

# Define the optimal policy
policy = np.empty((k, k), dtype=str)

def get_optimal_policy():
    for i in range(k):
        for j in range(k):
            state = (i, j)
            max_action = None
            max_value = float('-inf')
            for action in ["N", "S", "O", "E"]:
                next_state = state
                if action == "N":
                    next_state = (i - 1, j)
                elif action == "S":
                    next_state = (i + 1, j)
                elif action == "O":
                    next_state = (i, j - 1)
                elif action == "E":
                    next_state = (i, j + 1)

                if 0 <= next_state[0] < k and 0 <= next_state[1] < k:
                    action_value = rewards[next_state[0], next_state[1]] + gamma * values[next_state[0], next_state[1]]
                    if action_value > max_value:
                        max_value = action_value
                        max_action = action
            policy[i, j] = max_action
    return policy



def evaluation(compteur): #partie Evaluation de la Policy Iteration
    converge = False
    while converge == False : #1ere loop
        delta = 0
        #printGridworld()
        #print("\n")
        for ligne in range(k): #2e loop sur les états
            for colonne in range(k): #2e loop sur les états
                temp = values[ligne][colonne] #save old value v
                action = policy[ligne][colonne] #action en fonction de la politique
                next_state = (ligne, colonne)
                if next_state in [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (4, 4), (4, 5),
                                  (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11)]:
                    continue
                if action == "N":
                    next_state = (ligne - 1, colonne)
                elif action == "S":
                    next_state = (ligne + 1, colonne)
                elif action == "O":
                    next_state = (ligne, colonne - 1)
                elif action == "E":
                    next_state = (ligne, colonne + 1)
                if 0 <= next_state[0] < k and 0 <= next_state[1] < k:
                    compteur = compteur + 1
                    values[ligne][colonne] = gamma*values[next_state[0], next_state[1]] + rewards[next_state[0], next_state[1]]
                    delta = max(delta, abs(temp - values[ligne][colonne]))
        if delta < epsilon:
            converge = True
    #print(compteur)
    return compteur
            

    
def policyIteration(): #le reste de la Policy Iteration, la fonction à appeler
    matriceActionPrec = np.full((k, k),"N")
    converge = False
    compteur = 0
    while converge == False :
        compteur = evaluation(compteur)
        matriceActionNew = get_optimal_policy()
        converge = True
        for ligne in range(k):
            for colonne in range(k):
                if matriceActionPrec[ligne][colonne] != matriceActionNew[ligne][colonne]:
                    converge = False
        for ligne in range(k):
            for colonne in range(k):
                matriceActionPrec[ligne][colonne] = matriceActionNew[ligne][colonne]
    return matriceActionPrec




def printGridworld():
    for ligne in range(k - 1, 0, -1):
        for colonne in range(k):
            print(ligne, colonne, end=" ")
        print("\n")
        
# printGridworld()


####################### SARSA

#hyperparamètres v1 : alpha 0.1 gamma 0.9
#hyparamètres v2 : pareil que v1
alpha = 0.1
gamma = 0.9

#actions ordre : "N"(0), "S"(1), "O"(2), "E"(3)
listeActions = ["N", "S", "O", "E"]
#q = np.random.rand(k,k, 4)
q = np.zeros((k,k, 4))

rewards = np.zeros((k,k))
for i in range(0, k):
    for j in range(0, k):
        rewards[i, j] = -1
rewards[0, k - 1] = 2 * (k - 1)

version = 2
q[0, k - 1, 0] = 0
q[0, k - 1, 1] = 0
q[0, k - 1, 2] = 0
q[0, k - 1, 3] = 0
rewards[0, k - 1] = 2 * (k - 1)
etatsAbsorbant = [(0, k-1)]

if version == 2 :
    etatsAbsorbant = [(0, k-1), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (4, 4), (4, 5),
                      (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11)]
    for j in range(0, 8):
        rewards[8][j] = -2 * (k - 1)
        q[8, j, 0] = 0
        q[8, j, 1] = 0
        q[8, j, 2] = 0
        q[8, j, 3] = 0
    for j in range(4, 12):
        rewards[4][j] = -2 * (k - 1)
        q[4, j, 0] = 0
        q[4, j, 1] = 0
        q[4, j, 2] = 0
        q[4, j, 3] = 0
        
#print(rewards)

def egreedy(s, epsilon):
    indiceActionMax = np.argmax(q[s])
    aMax = listeActions[indiceActionMax]
    listeProbas = [epsilon/4,epsilon/4,epsilon/4,epsilon/4]
    listeProbas[indiceActionMax] = (1-epsilon)+(epsilon/4)
    action = np.random.choice(listeActions, p=listeProbas) 
    return action

"""def egreedy(state, epsilon):
    if np.random.rand() < epsilon:
        temp = np.random.choice(4)
        return listeActions[temp]
    else:
        temp = np.argmax(q[state])
        return listeActions[temp]"""

def get_next_state(s,action):
    next_s = (s[0], s[1])
    if action == "N":
        next_s = (s[0] - 1, s[1])
    elif action == "S":
        next_s = (s[0] + 1, s[1])
    elif action == "O":
        next_s = (s[0], s[1] - 1)
    elif action == "E":
        next_s = (s[0], s[1] + 1)
    return next_s

def verif_state(s, action, next_s, epsilon):
    while not (0 <= next_s[0] < k and 0 <= next_s[1] < k):
        action = egreedy(s, epsilon)
        next_s = get_next_state(s, action)
    return next_s, action

def sarsa(etatsAbsorbant):
    epsilon = 1
    for episode in range(0,1000):
        t = 0
        s = (11, 0) #init state s 
        action = egreedy(s, epsilon) #choose a from s (egreedy)
        print(episode)
        while s not in etatsAbsorbant: #2e boucle
            print(s, action)
            t = t +1
            epsilon = 1/t
            next_s = get_next_state(s, action) #s'
            #verif si s' est bien dans les limites, sinon recommence... :
            next_s, action = verif_state(s, action, next_s, epsilon) 
            
            next_action = egreedy(next_s, epsilon) #choose a' from s' (egreedy)
            next_next_s = get_next_state(next_s, next_action) #s''
            #verif si s'' est bien dans les limites, sinon recommence... :
            next_next_s, next_action = verif_state(next_s, next_action, next_next_s, epsilon)
            
            #print(s[0], s[1], next_s[0], next_s[1])
            q[s[0], s[1],listeActions.index(action)] = (1 - alpha)*q[s[0], s[1], listeActions.index(action)] + alpha*(rewards[next_s[0], next_s[1]] + gamma*q[next_s[0], next_s[1], listeActions.index(next_action)])
            s = next_s
            action = next_action
            
        #pour actualiser q sur les états absorbant
        """for i in range(3):
            q[s[0], s[1],listeActions.index(action)] = (1 - alpha)*q[s[0], s[1], listeActions.index(action)] + alpha*(rewards[next_s[0], next_s[1]] + gamma*q[next_s[0], next_s[1], listeActions.index(next_action)])
"""


"""def optimalPolicySARSA():
    optimal_policy = np.empty((k, k), dtype=str)
    
    for i in range(k):
        for j in range(k):
            state = (i, j)
            print(np.argmax(q[state[0], state[1]]))
            optimal_action = ['N', 'S', 'O', 'E'][np.argmax(q[state[0], state[1]])]
            optimal_policy[i, j] = optimal_action
    return optimal_policy"""


def optimalPolicySARSA():
    optimal_policy = np.empty((k, k), dtype=str)
    
    for i in range(k):
        for j in range(k):
            state = (i, j)
            max_action = None
            max_value = float('-inf')
            for action in ["N", "S", "O", "E"]:
                next_state = get_next_state(state, action)
                if 0 <= next_state[0] < k and 0 <= next_state[1] < k:
                    action_value = q[state[0], state[1], listeActions.index(action)]
                    if action_value > max_value:
                        max_value = action_value
                        max_action = action
            
            optimal_policy[i, j] = max_action
    return optimal_policy

"""sarsa(etatsAbsorbant)
print(optimalPolicySARSA())        
print(q)"""



########### Q LEARNING
    
#hyperparamètres pour v1 : alpha 0.1 gamma 0.9
#hyperparamètres pour v2 : alpha 0.2 gamma 0.9

alpha = 0.2
gamma = 0.9

#actions ordre : "N"(0), "S"(1), "O"(2), "E"(3)
listeActions = ["N", "S", "O", "E"]

#q = np.random.rand(k,k, 4)
q = np.zeros((k,k, 4))

rewards = np.zeros((k,k))
for i in range(0, k):
    for j in range(0, k):
        rewards[i, j] = -1
rewards[0, k - 1] = 2 * (k - 1)

version = 2
q[0, k - 1, 0] = 0
q[0, k - 1, 1] = 0
q[0, k - 1, 2] = 0
q[0, k - 1, 3] = 0
rewards[0, k - 1] = 2 * (k - 1)
etatsAbsorbant = [(0, k-1)]

if version == 2 :
    etatsAbsorbant = [(0, k-1), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (4, 4), (4, 5),
                      (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11)]
    for j in range(0, 8):
        rewards[8][j] = -2 * (k - 1)
        q[8, j, 0] = 0
        q[8, j, 1] = 0
        q[8, j, 2] = 0
        q[8, j, 3] = 0
    for j in range(4, 12):
        rewards[4][j] = -2 * (k - 1)
        q[4, j, 0] = 0
        q[4, j, 1] = 0
        q[4, j, 2] = 0
        q[4, j, 3] = 0
        
#print(rewards)

def egreedy(s, epsilon):
    indiceActionMax = np.argmax(q[s])
    aMax = listeActions[indiceActionMax]
    listeProbas = [epsilon/4,epsilon/4,epsilon/4,epsilon/4]
    listeProbas[indiceActionMax] = (1-epsilon)+(epsilon/4)
    action = np.random.choice(listeActions, p=listeProbas) 
    return action

"""def egreedy(state, epsilon):
    if np.random.rand() < epsilon:
        temp = np.random.choice(4)
        return listeActions[temp]
    else:
        temp = np.argmax(q[state])
        return listeActions[temp]"""

def get_next_state(s,action):
    next_s = (s[0], s[1])
    if action == "N":
        next_s = (s[0] - 1, s[1])
    elif action == "S":
        next_s = (s[0] + 1, s[1])
    elif action == "O":
        next_s = (s[0], s[1] - 1)
    elif action == "E":
        next_s = (s[0], s[1] + 1)
    return next_s

def verif_state(s, action, next_s, epsilon):
    while not (0 <= next_s[0] < k and 0 <= next_s[1] < k):
        action = egreedy(s, epsilon)
        next_s = get_next_state(s, action)
    return next_s, action

def qlearning(etatsAbsorbant):
    epsilon = 1
    for episode in range(0,1000):
        t = 0
        s = (11, 0) #init state s 
        print(episode)
        while s not in etatsAbsorbant: #2e boucle
            action = egreedy(s, epsilon) #choose a from s (egreedy)
            print(s, action)
            t = t +1
            epsilon = 1/t
            next_s = get_next_state(s, action) #s'
            #verif si s' est bien dans les limites, sinon recommence... :
            next_s, action = verif_state(s, action, next_s, epsilon) 
                
            max_action = None
            max_value = float('-inf')
            for action_tmp in ["N", "S", "O", "E"]:
                next_next_state = get_next_state(next_s, action_tmp)
                if 0 <= next_next_state[0] < k and 0 <= next_next_state[1] < k :
                    action_value = q[next_next_state[0], next_next_state[1], listeActions.index(action_tmp)]
                    if action_value > max_value:
                        max_value = action_value
                        max_action = action_tmp
            
            #print(s[0], s[1], next_s[0], next_s[1])
            q[s[0], s[1],listeActions.index(action)] = (1 - alpha)*q[s[0], s[1], listeActions.index(action)]+ alpha*(rewards[next_s[0], next_s[1]] + gamma*q[next_s[0], next_s[1], listeActions.index(max_action)])
            s = next_s
            



def optimalPolicyQlearning():
    optimal_policy = np.empty((k, k), dtype=str)
    
    for i in range(k):
        for j in range(k):
            state = (i, j)
            max_action = None
            max_value = float('-inf')
            for action in ["N", "S", "O", "E"]:
                next_state = get_next_state(state, action)
                if 0 <= next_state[0] < k and 0 <= next_state[1] < k:
                    action_value = q[state[0], state[1], listeActions.index(action)]
                    if action_value > max_value:
                        max_value = action_value
                        max_action = action
            
            optimal_policy[i, j] = max_action
    return optimal_policy

"""qlearning(etatsAbsorbant)
print(optimalPolicyQlearning())        
print(q)"""



####### Monte Carlo :
    
    
listeEtats = []
returns = {}
for i in range(k):
    for j in range(k):
        listeEtats.append((i,j))
        for a in range(len(listeActions)):
            returns["("+str(i)+","+str(j)+")"+listeActions[a]] = []
#print(returns)
#print(listeEtats)
    
def exploringStarts():
    etat = np.random.choice(len(listeEtats),1)
    action = np.random.choice(listeActions)
    #print(listeEtats[etat[0]])
    while listeEtats[etat[0]] in etatsAbsorbant:
        etat = np.random.choice(len(listeEtats),1)
    #print(listeEtats[etat[0]], action)
    return listeEtats[etat[0]], action
    
def get_next_state_mc(s,action):
    next_s = (s[0], s[1])
    if action == "N":
        next_s = (s[0] - 1, s[1])
    elif action == "S":
        next_s = (s[0] + 1, s[1])
    elif action == "O":
        next_s = (s[0], s[1] - 1)
    elif action == "E":
        next_s = (s[0], s[1] + 1)
    return next_s
    
def verif_state_mc(s, action):
    next_s = get_next_state_mc(s, action)
    while not (0 <= next_s[0] < k and 0 <= next_s[1] < k):
        _, action = exploringStarts()
        next_s = get_next_state_mc(s, action)
    return s, next_s, action
    


listeActions = ["N", "S", "O", "E"]
q = np.zeros((k,k, 4))
gamma = 0.9

#optimal_policy = np.empty((k, k), dtype=str)
#optimal_policy= np.random.choice(listeActions)
optimal_policy = np.random.choice(listeActions, size=(k, k))

for i in range(k) :
    for j in range(k) :
        etat = [i,j]
        etat, next_etat, action = verif_state_mc(etat, optimal_policy[etat[0], etat[1]])
        optimal_policy[etat[0], etat[1]] = action
        

rewards = np.zeros((k,k))
for i in range(0, k):
    for j in range(0, k):
        rewards[i, j] = -1
        
rewards[0, k - 1] = 2 * (k - 1)

version = 2
q[0, k - 1, 0] = 0
q[0, k - 1, 1] = 0
q[0, k - 1, 2] = 0
q[0, k - 1, 3] = 0
rewards[0, k - 1] = 2 * (k - 1)
etatsAbsorbant = [(0, k-1)]

if version == 2 :
    etatsAbsorbant = [(0, k-1), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (4, 4), (4, 5),
                      (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11)]
    for j in range(0, 8):
        rewards[8][j] = -2 * (k - 1)
        q[8, j, 0] = 0
        q[8, j, 1] = 0
        q[8, j, 2] = 0
        q[8, j, 3] = 0
    for j in range(4, 12):
        rewards[4][j] = -2 * (k - 1)
        q[4, j, 0] = 0
        q[4, j, 1] = 0
        q[4, j, 2] = 0
        q[4, j, 3] = 0




def optimal_policy_mc(st):
    max_action = None
    max_value = float('-inf')
    for action in ["N", "S", "O", "E"]:
        next_state = get_next_state(st, action)
        if 0 <= next_state[0] < k and 0 <= next_state[1] < k:
            action_value = q[st[0], st[1], listeActions.index(action)]
            if action_value > max_value:
                max_value = action_value
                max_action = action
    
    optimal_policy[st[0], st[1]] = max_action
    



#marche v1 avec gamma 0.9 et 1000 ite
#marche v2 en augmentant ite à 10000

def monteCarlo():
        
    for episode in range(0,10000):
        print(episode)
        count = 0
        
        listeS = []
        listeA = []
        listeR = []
        etat, action = exploringStarts()
        etat, next_s, action = verif_state_mc(etat, action)
        listeS.append(etat)
        listeA.append(action)
        listeR.append(rewards[etat[0], etat[1]])
        s = next_s
        
        
        while s not in etatsAbsorbant and count < 144:
            listeS.append(s)
            count = count + 1
            
            #action = np.random.choice(listeActions)
            action = optimal_policy[s[0], s[1]]
            
            #s, next_s, action = verif_state_mc(s, action)
            next_s = get_next_state_mc(s, action)
            
            listeA.append(action)
            listeR.append(rewards[etat[0], etat[1]])
            s = next_s
        listeR.append(rewards[s[0], s[1]])
        
        
        """print(listeS, listeA, listeR)
        print(s)"""
        
        g = 0
        T = len(listeS)
        
        """print(len(listeS))
        print(len(listeA))
        print(len(listeR))"""
        
        for t in range(T-1, -1, -1):
            #print(t, listeS[t])
            g = gamma*g + listeR[t+1]
            st = listeS[t]
            print(listeS[0:t])
            if st not in listeS[0:t] :
                listeReturns = returns["("+str(st[0])+","+str(st[1])+")"+listeA[t]]
                listeReturns.append(g)
                returns["("+str(st[0])+","+str(st[1])+")"+listeA[t]] = listeReturns
                #print("q", q[st[0], st[1]], listeActions.index(listeA[t]))
                q[st[0], st[1], listeActions.index(listeA[t])] = np.mean(returns["("+str(st[0])+","+str(st[1])+")"+listeA[t]])
                optimal_policy_mc(st)
     
# monteCarlo()
# print(optimal_policy)





# Print the optimal values and policy
# print("Optimal Values:")
# print(values)
print("Optimal Policy:")
#printGridworld()
valueIteration()
print(get_optimal_policy())

#print(policyIteration())
