import numpy as np

# Define the parameters of the MDP
n = 12
m = 12

# Initial state
start_state = (0, 0)

# Reward matrix
rewards = np.zeros((n, m))
for i in range(0, n):
    for j in range(0, m):
        rewards[i, j] = -1
rewards[n - 1, m - 1] = 2 * (n - 1)
for j in range(0, 8):
    rewards[3][j] = -2 * (n - 1)
for j in range(4, 12):
    rewards[7][j] = -2 * (n - 1)

gamma = 0.9

print(rewards)

# Stopping condition for value iteration
epsilon = 0.001

# Initialize the values of all states to 0
values = np.zeros((n, m))

# Apply the value iteration algorithm
while True:
    delta = 0
    for i in range(0, n):
        for j in range(0, m):
            state = (i, j)
            max_value = float('-inf')
            for action in ["N", "S", "O", "E"]:
                next_state = state
                if next_state in [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (7, 4), (7, 5),
                                  (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11)]:
                    continue
                if action == "N":
                    next_state = (i - 1, j)
                elif action == "S":
                    next_state = (i + 1, j)
                elif action == "O":
                    next_state = (i, j - 1)
                elif action == "E":
                    next_state = (i, j + 1)

                if 0 <= next_state[0] < n and 0 <= next_state[1] < m:
                    action_value = rewards[next_state[0], next_state[1]] + gamma * values[next_state[0], next_state[1]]
                    if action_value > max_value:
                        max_value = action_value

            delta = max(delta, abs(values[i, j] - max_value))
            values[i, j] = max_value

    if delta < epsilon:
        break

# Define the optimal policy
policy = np.empty((n, m), dtype=str)


def get_optimal_policy():
    for i in range(n):
        for j in range(m):
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

                if 0 <= next_state[0] < n and 0 <= next_state[1] < m:
                    action_value = rewards[next_state[0], next_state[1]] + gamma * values[next_state[0], next_state[1]]
                    if action_value > max_value:
                        max_value = action_value
                        max_action = action
            policy[i, j] = max_action
    return policy


def printGridworld(policy):
    for ligne in range(n - 1, 0, -1):
        for colonne in range(n):
            print(policy[ligne, colonne], end=" ")
        print("\n")


# Print the optimal values and policy
# print("Optimal Values:")
# print(values)
print("Optimal Policy:")
# printGridworld()
print(get_optimal_policy())
