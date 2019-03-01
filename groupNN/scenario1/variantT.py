# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
import csv
import numpy as np
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../groupNN')
from testcharacter import TestCharacter

# Create the game
# random.seed(123) # TODO Change this if you want different random choices

Q_table = {}
weights = 5.0 * np.random.randn(12, 1)
epsilon = 1.0
alpha = 0.5

for i in range(0, 5000):
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("monster", # name
                                "S",       # avatar
                                3, 5,      # position
    ))
    g.add_monster(StupidMonster("monster", # name
                                        "A",       # avatar
                                        3, 13#,     # position
                                       # 2          # detection range
    ))

    my_char = TestCharacter("me", "C",  0, 18, alpha, epsilon, Q_table, weights)
    g.add_character(my_char)

    # Run!
    g.go(wait=1)
    Q_table = my_char.Q_table
    weights = my_char.weights
    epsilon = my_char.epsilon
    alpha = my_char.alpha

with open('q_table.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    for k, v in Q_table:
        writer.writerow([k] + v)
writeFile.close()