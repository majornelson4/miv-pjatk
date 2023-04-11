import numpy as np
import matplotlib.pyplot as plt

states = {
    'Paper': 0,
    'Scissors': 1,
    'Rock': 2
}
wallet = 0
number_of_plays = 10000
# transition_matrix = np.array([[2, 3, 1], [2, 2, 4], [2, 3, 1]])
transition_matrix = np.eye(3, 3, dtype='int8')
opponent = np.array([0, 1, 0])  # paper-scissors-rock


def update_wallet(pl_state, opponent_state):
    if opponent_state == pl_state:
        return 0
    combinations = {'Scissors': 'Paper',
                    'Rock': 'Scissors',
                    'Paper': 'Rock'
                    }  # win combinations
    if combinations[pl_state] != opponent_state:
        return -1
    else:
        return 1


state = 'Paper'
for index in range(number_of_plays):
    opponent_next_move_pred = np.random.choice(list(states.keys()), p=transition_matrix[states.get(state)] / sum(
        transition_matrix[states.get(state)]))
    state = np.random.choice(list(states.keys()), p=opponent)
    transition_matrix[states.get(state)][states.get(opponent_next_move_pred)] += 1
    wallet += update_wallet(pl_state=state, opponent_state=opponent_next_move_pred)
    plt.plot(index, wallet, 'b*')
    state = opponent_next_move_pred  # changing current state to opponent
    print('Wallet: ' + str(wallet))

plt.draw()
plt.xlabel("Number of steps")
plt.ylabel("Wallet")
plt.show()
