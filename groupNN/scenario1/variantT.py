# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
import csv
import numpy as np
import math
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../groupNN')
from testcharacter import TestCharacter

# Create the game
# random.seed(123) # TODO Change this if you want different random choices

Q_table = {}
actions_taken = [0] * 10
weights = 10.0 * np.random.randn(1, 11)
prev_weights = np.zeros_like(weights)
cut_off = 0.00000000000001
weights_diff = math.inf
epsilon = 1.0
alpha = 0.00025
c = 0
minY = 18
open('log.txt', 'w').close()
while(c < 10000 and epsilon > 0 and (cut_off < weights_diff or c < 100)):
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("monster", # name
                                "S",       # avatar
                                random.randint(0,7), random.randint(0, 18),      # position
    ))

    my_char = TestCharacter("me", "C",  random.randint(0, 7), random.randint(0, 18), alpha, epsilon, Q_table, weights, actions_taken)
    g.add_character(my_char)

    # Run!
    final_score = g.go(wait=1)
    f = open("log.txt", "a")
    f.write(str(final_score - my_char.prev_score) + "\n")
    f.close()
    Q_sa = np.dot(my_char.weights, np.array(my_char.prev_state.to_vector())).item(0)
    my_char.Q_table[''.join(str(e) for e in my_char.prev_state.to_vector())][my_char.prev_action] = Q_sa
    my_char.update_weights(my_char.prev_state.to_vector(), final_score - my_char.prev_score, Q_sa)
    Q_table = my_char.Q_table
    weights = my_char.weights
    epsilon = my_char.epsilon
    alpha = my_char.alpha
    actions_taken = my_char.actions_taken
    weights_diff = abs(np.sum(weights - prev_weights))
    np.copyto(prev_weights, weights)
    c += 1
    if c % 15 == 0:
        minY -= 1

# with open('q_table.csv', 'w') as writeFile:
    # fields = ['states', 'q_vals']
    # writer = csv.DictWriter(writeFile, fieldnames=fields)
    # writer.writeheader()
    # writer.writerows(Q_table)
# writeFile.close()

print(Q_table)
print(len(Q_table))
print("c: " + str(c < 1000))
print("e: " + str(epsilon > 0))
print("w: " + str(cut_off < weights_diff))