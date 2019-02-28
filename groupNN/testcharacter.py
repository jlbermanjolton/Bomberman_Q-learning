# This is necessary to find the main code
import random
import sys

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import numpy as np
import tflearn
import tensorflow as tf
from tflearn import conv_2d, fully_connected, input_data

from sensed_world import SensedWorld

entities = ['wall', 'hero', 'enemy', 'bomb', 'explosion', 'monster', 'exit']
actions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'bomb', 'nothing']

# TODO Save and Load Q_Table from file for persistence (optionally human readable ie JSON)
Q_Table = []
previous_state = None
previous_action = None
previous_reward = None


class TestCharacter(CharacterEntity):

    def do(self, wrld):
        # TODO Create a Rewards Table and update the previous reward

        # TODO Update the Q_Table based on previous state-action-reward

        # Obtain the state of the current world
        world_cpy = SensedWorld.from_world(wrld)

        # Construct a grid of all possible states of the world
        # This consists of an X * Y * entities array
        # In the sample game this is 8x19x7 resulting in a state space of 1.86*10^137 (TOO LARGE)
        grid = []
        for x in range(0, world_cpy.width()):
            column = []
            for y in range(0, world_cpy.height()):
                column.append([0] * len(entities))
                # Check if Wall in location, if so flip corresponding position
                if world_cpy.grid[x][y]:
                    column[y][entities.index("wall")] = 1
            grid.append(column)

        # Add Characters from world to grid
        for i, character_list in world_cpy.characters.items():
            for character in character_list:
                if character == world_cpy.me(self):  # Found current player
                    grid[character.x][character.y][entities.index("hero")] = 1
                else:
                    grid[character.x][character.y][entities.index("enemy")] = 1

        # Add bombs from world to grid
        for bomb in world_cpy.bombs.items():
            grid[bomb[1].x][bomb[1].y][entities.index("bomb")] = 1

        # Add explosions from world to grid
        for explosion in world_cpy.explosions.items():
            grid[explosion[1].x][explosion[1].y][entities.index("explosion")] = 1

        # Add monsters from world to grid
        for i, monster_list in world_cpy.monsters.items():
            for monster in monster_list:
                grid[monster.x][monster.y][entities.index("monster")] = 1

        # Add exit cell from world to grid
        if world_cpy.exitcell is not None:
            grid[world_cpy.exitcell[0]][world_cpy.exitcell[1]][entities.index("exit")] = 1

        # Get the current action score pairs of the current world state
        action_score_pairs = None
        found = False
        for entry in Q_Table:
            if entry[0] == grid:
                action_score_pairs = entry[1]
                found = True
                break


        # If the state is not currently in the QTable, add it with 0's as Q values for all actions
        # TODO is propagating new values as zero the best strategy?
        if not found:
            action_score_pairs = [[0] * len(actions)]
            Q_Table.append([grid, action_score_pairs])

        # Using epsilon greedy formula, choose an action based on the obtained action score pairs
        # TODO implement epsilon greedy, using random currently
        action_selection = random.randint(0, len(actions) - 1)

        # Parse action selection and perform action
        action_selection_string = actions[action_selection]
        if action_selection_string == 'north':
            self.move(0, -1)
        elif action_selection_string == 'south':
            self.move(0, 1)
        elif action_selection_string == 'east':
            self.move(1, 0)
        elif action_selection_string == 'west':
            self.move(-1, 0)
        elif action_selection_string == 'bomb':
            self.place_bomb()

        # Update the persistent variables to hold information on this action selection
        # Reward is unknown until action is completed ie start of this function
        previous_state = grid
        previous_action = action_selection
        previous_reward = None

        # Profiling
        memory_usage = sys.getsizeof(Q_Table)

        pass



# =============================
#   TFLearn Deep Q Network
# =============================
def build_dqn(num_actions, action_repeat):
    """
    Building a DQN.
    """
    networkInput = tflearn.input_data(shape=[None, 8, 19, len(entities)])
    conv = conv_2d(networkInput, 2, 4, activation='leaky_relu')
    conv2 = conv_2d(conv, 4, 4, activation='leaky_relu')
    conv3 = conv_2d(conv, 8, 4, activation='leaky_relu')
    fullyConnected = tflearn.fully_connected(conv3, 8 * 4, activation='leaky_relu')
    qValues = tflearn.fully_connected(fullyConnected, len(actions), activation='leaky_relu')
    return networkInput, qValues
