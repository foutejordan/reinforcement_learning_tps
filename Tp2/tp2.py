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



def iteValeurBis(compteur): #partie Evaluation de la Policy Iteration
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
            

    
def getPolicyBis(): #le reste de la Policy Iteration, la fonction à appeler
    matriceActionPrec = np.full((k, k),"N")
    converge = False
    compteur = 0
    while converge == False :
        compteur = iteValeurBis(compteur)
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


# Print the optimal values and policy
# print("Optimal Values:")
# print(values)
print("Optimal Policy:")
#printGridworld()
valueIteration()
print(get_optimal_policy())

#print(getPolicyBis())
