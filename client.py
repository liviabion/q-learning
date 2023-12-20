#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
import random as rd
import numpy as np

s = cn.connect(2037)
alpha = 0.01

# Bellman's Formula: U(s) = R(s) + γ maxa Σs’ T(s,a,s’)
def qValue(next_st, rec):
    gamma = 0.5
    # Calculate the value using the Bellman equation
    value = rec + gamma * max(qTable[next_st])
    return value

# Choose the best move based on the current state
def bestMove(state):
    # Determine the index of the maximum Q-value for the given state
    if qTable[state, 0] > qTable[state, 1] and qTable[state, 0] > qTable[state, 2]:
        return 0
    elif qTable[state, 1] > qTable[state, 0] and qTable[state, 1] > qTable[state, 2]:
        return 1
    else:
        return 2


qTable = np.loadtxt('resultado.txt')
np.set_printoptions(precision=6)

# Initialize current state, current reward, possible actions, and exploration parameter
curr_state = 0
curr_reward = -14
actions = ["left", "right", "jump"]
epsilon = 0

# Main training loop
while True:
    # Print the current state
    print(curr_state)

    # Choose an action based on epsilon-greedy strategy
    if rd.random() < epsilon:
        move = actions[rd.randint(0, 2)]  # Choose a random action
        print(f'Random action chosen for state {curr_state}: {move}')
    else:
        move = actions[bestMove(curr_state)]
        print(f'Best action chosen for state {curr_state}: {move}')

    # Map action to column index
    if move == "left":
        moveColumn = 0
    elif move == "right":
        moveColumn = 1
    else:
        moveColumn = 2

    # Get the next state and reward from the environment
    state, reward = cn.get_state_reward(s, move)
    # Extract relevant information from the state
    state = state[2:]

    # Convert binary state and direction to decimal
    state = int(state, 2)
    next_state = state

    # Display the previous and new Q-values for the chosen action
    print(f'Previous value of this action: {qTable[curr_state][moveColumn]}')
    qTable[curr_state][moveColumn] = qTable[curr_state][moveColumn] + alpha * (qValue(next_state, curr_reward) - qTable[curr_state][moveColumn])
    print(f'New value of this action: {qTable[curr_state][moveColumn] + alpha * (qValue(next_state, curr_reward) - qTable[curr_state][moveColumn])}')

    # Update the current state and reward for the next iteration
    curr_state = next_state
    curr_reward = reward

    # Save the updated Q-table to a file
    np.savetxt('resultado.txt', qTable, fmt="%f")
