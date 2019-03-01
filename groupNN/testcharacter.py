# This is necessary to find the main code
import random
import sys

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import math

import numpy as np
import tflearn
import tensorflow as tf
from tflearn import conv_2d, fully_connected, input_data

from sensed_world import SensedWorld

entities = ['wall', 'hero', 'enemy', 'bomb', 'explosion', 'monster', 'exit']
actions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'bomb', 'wait']

# TODO Save and Load Q_Table from file for persistence (optionally human readable ie JSON)
Q_Table = []
previous_state = None
previous_action = None
previous_reward = None

learning_rate = 0.001
# Reward discount rate
gamma = 0.99
# Number of timesteps to anneal epsilon
anneal_epsilon_timesteps = 400000

checkpoint_path = 'qlearning.tflearn.ckpt'
checkpoint_interval = 2000



class TestCharacter(CharacterEntity):

    def do(self, wrld):
        pass

