import numpy as np

# Define the parameters of the MDP
k = 12
version = 2 #version 1 ou 2

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


print(rewards)

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
        
  


####################### SARSA

alpha = 0.5
#actions ordre : "N"(0), "S"(1), "O"(2), "E"(3)
listeActions = ["N", "S", "O", "E"]
q = np.random.rand(k,k, 4)
rewards = np.zeros((k,k))
for i in range(0, k):
    for j in range(0, k):
        rewards[i, j] = -1
rewards[0, k - 1] = 2 * (k - 1)
version = 1
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

def sarsa(etatsAbsorbant):
    epsilon = 0.1
    for episode in range(0,100):
        s = (11, 0)
        action = egreedy(s, epsilon)
        
        while s not in etatsAbsorbant:
            if action == "N":
                next_s = (s[0] - 1, s[1])
            elif action == "S":
                next_s = (s[0] + 1, s[1])
            elif action == "O":
                next_s = (s[0], s[1] - 1)
            elif action == "E":
                next_s = (s[0], s[1] + 1)
            if 0 <= next_s[0] < k and 0 <= next_s[1] < k:
                next_action = egreedy(s, epsilon)
                print(s[0], s[1], next_s[0], next_s[1])
                q[s[0], s[1],listeActions.index(action)] = (1 - alpha)*q[s[0], s[1], listeActions.index(action)] + alpha*(rewards[next_s[0], next_s[1]] + q[next_s[0], next_s[1], listeActions.index(next_action)])
                s = next_s
                action = next_action

optimal_policy = np.empty((k, k), dtype=str)

for i in range(k):
    for j in range(k):
        state = (i, j)
        optimal_action = ['N', 'S', 'O', 'E'][np.argmax(q[state[0], state[1]])]
        optimal_policy[i, j] = optimal_action

sarsa(etatsAbsorbant)
print(optimal_policy)        
        


# Print the optimal values and policy
# print("Optimal Values:")
# print(values)
print("Optimal Policy:")
#printGridworld()
"""valueIteration()
print(get_optimal_policy())"""

#print(policyIteration())
