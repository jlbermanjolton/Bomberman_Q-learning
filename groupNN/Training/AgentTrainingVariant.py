# This is necessary to find the main code
import random
import sys
import os

sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
#from game import Game
from AgentTraining import Game

sys.path.insert(1, '../groupNN')
from testcharacter import TestCharacter
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

import numpy as np
import tflearn
import tensorflow as tf
from tflearn import conv_2d, fully_connected, input_data

def build_dqn():
    """
    Building a DQN.
    """
    networkInput = tflearn.input_data(shape=[None, 8, 19, len(entities)])
    conv = conv_2d(networkInput, 2, 4, activation='leaky_relu')
    conv2 = conv_2d(conv, 4, 4, activation='leaky_relu')
    conv3 = conv_2d(conv, 8, 4, activation='leaky_relu')
    fullyConnected = tflearn.fully_connected(conv3, 8 * 4, activation='leaky_relu')
    #fullyConnected2 = tflearn.fully_connected(fullyConnected, 8 * 4, activation='leaky_relu')
    #fullyConnected3 = tflearn.fully_connected(fullyConnected2, 8 * 4, activation='leaky_relu')

    qValues = tflearn.fully_connected(fullyConnected, len(actions), activation='leaky_relu')
    return networkInput, qValues


def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def build_graph():
    # Create shared deep q network
    shared_inputs, q_network = build_dqn()
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network
    shared_target_inputs, target_q_network = build_dqn()
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network

    # Op for periodically updating target network with online network weights
    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, len(actions)])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"shared_inputs": shared_inputs,
                 "q_values": q_values,
                 "shared_target_inputs": shared_target_inputs,
                 "target_q_values": target_q_values,
                 "reset_target_network_params": reset_target_network_params,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}

    return graph_ops


def get_world_state(game):
    # Obtain the state of the current world
    world_cpy = game.world

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
            if character.name == "me":  # Found current player
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

    return grid


####################################################################
#                          Begin Agent                             #
####################################################################


entities = ['wall', 'hero', 'enemy', 'bomb', 'explosion', 'monster', 'exit']
actions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'bomb', 'nothing']

learning_rate = 0.01
# Reward discount rate
gamma = 0.99
# Number of timesteps to anneal epsilon
anneal_epsilon_timesteps = 40000
# Async gradient update frequency of each learning thread
I_AsyncUpdate = 5
# Timestep to reset the target network
I_target = 40000

checkpoint_path = 'qlearning.tflearn.ckpt'
checkpoint_interval = 2000

epochs = 2000

# Initialize network gradients
state_batch = []
action_batch = []
y_batch = []

#final_epsilon = sample_final_epsilon()
final_epsilon = 0.2
initial_epsilon = 1.0
epsilon = 0.2

graph_ops = build_graph()  # TODO load graph from saver
saver = tf.train.Saver(max_to_keep=5)

# Unpack the graph operations
shared_inputs = graph_ops["shared_inputs"]
q_values = graph_ops["q_values"]
shared_target_inputs = graph_ops["shared_target_inputs"]
target_q_values = graph_ops["target_q_values"]
reset_target_network_params = graph_ops["reset_target_network_params"]
a = graph_ops["a"]
y = graph_ops["y"]
grad_update = graph_ops["grad_update"]



with tf.Session() as session:
    # Initialize variables
    session.run(tf.initialize_all_variables())

    # Specify the name of the checkpoints directory
    checkpoint_dir = "checkpoints"

    # Create the directory if it does not already exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Specify the path to the checkpoint file
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.ckpt")

    # Initialize Variables
    if tf.train.checkpoint_exists(checkpoint_file):
        print("Restoring from file: ", checkpoint_file)
        saver.restore(session, checkpoint_file)
    else:
        print("Initializing from scratch")
        session.run(tf.global_variables_initializer())

    total_step_count = 0

    for i in range(0, epochs):

        # Create the game
        g = Game.fromfile('map.txt')

        g.add_character(TestCharacter("me",  # name
                                      "C",  # avatar
                                      7, 18  # position
                                      ))
        # g.add_monster(SelfPreservingMonster("monster",  # name
        #                                     "A",  # avatar
        #                                     3, 13,  # position
        #                                     2  # detection range
        #                                     ))
        g.add_monster(StupidMonster("monster",  # name
                                    "S",  # avatar
                                    0, 10,  # position
                                    ))
        # g.add_monster(StupidMonster("monster",  # name
        #                             "S",  # avatar
        #                             3, 13,  # position
        #                             ))

        g.init_GUI()

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_step_count = 0

        while not g.done():
            # Get initial game observation
            world_state = get_world_state(g)

            # Forward the deep q network, get Q(s,a) values
            readout_t = q_values.eval(session=session, feed_dict={shared_inputs: [world_state]})

            # Choose next action based on e-greedy policy
            action_taken = np.zeros([len(actions)])  # One hot encoding
            if random.random() <= epsilon:
                action_index = random.randrange(len(actions))
            else:
                action_index = np.argmax(readout_t)
            action_taken[action_index] = 1

            # Scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps

            # Parse action selection and perform action
            action_selection_string = actions[action_index]

            # Find Character indicies in the world lists
            char_index = -1
            char_inner_index = -1

            for i, character_list in g.world.characters.items():
                for j, character in enumerate(character_list):
                    if character.name == "me":  # Found current player
                        char_index = i
                        char_inner_index = j

            if char_index == -1:
                print("Agent not found in world, terminating session")
                break

            if action_selection_string == 'N':
                g.world.characters[char_index][char_inner_index].move(0, -1)
            elif action_selection_string == 'NE':
                g.world.characters[char_index][char_inner_index].move(1, -1)
            elif action_selection_string == 'NW':
                g.world.characters[char_index][char_inner_index].move(-1, -1)
            elif action_selection_string == 'S':
                g.world.characters[char_index][char_inner_index].move(0, 1)
            elif action_selection_string == 'SE':
                g.world.characters[char_index][char_inner_index].move(1, 1)
            elif action_selection_string == 'SW':
                g.world.characters[char_index][char_inner_index].move(-1, 1)
            elif action_selection_string == 'E':
                g.world.characters[char_index][char_inner_index].move(1, 0)
            elif action_selection_string == 'W':
                g.world.characters[char_index][char_inner_index].move(-1, 0)
            elif action_selection_string == 'wait':
                g.world.characters[char_index][char_inner_index].move(0, 0)
            elif action_selection_string == 'bomb':
                g.world.characters[char_index][char_inner_index].place_bomb()

            # Update the world and obtain the reward from the actions
            value = g.world.scores["me"]
            g.next_step(wait=1)
            statePrime = get_world_state(g)
            valuePrime = g.world.scores["me"]
            reward = valuePrime - value

            # Accumulate gradients
            readout_j1 = target_q_values.eval(session=session, feed_dict={shared_target_inputs: [statePrime]})

            # Scale rewards to between -1 and 1
            #clipped_r_t = np.clip(reward, -1, 1)
            #y_batch.append(clipped_r_t + gamma * np.max(readout_j1))  # Use Bellman
            y_batch.append(reward + gamma * np.max(readout_j1))  # Use Bellman

            action_batch.append(action_taken)
            state_batch.append(world_state)

            # Update counters
            total_step_count += 1
            ep_step_count += 1
            ep_reward += reward
            episode_ave_max_q += np.max(readout_t)

            # Optionally update target network
            if total_step_count % I_target == 0:
                session.run(reset_target_network_params)

            # Optionally update online network
            if total_step_count % I_AsyncUpdate == 0 or g.done():
                if state_batch:
                    session.run(grad_update, feed_dict={y: y_batch,
                                                        a: action_batch,
                                                        shared_inputs: state_batch})
                # Clear gradients
                state_batch = []
                action_batch = []
                y_batch = []

            # Save the model
            if total_step_count % checkpoint_interval == 0:
                #saver.save(session, "qlearning.ckpt", global_step=total_step_count)
                saver.save(session, checkpoint_file)

        g.deinit_GUI()

        print("Completed all iterations")

        # Run!
        # batch_data, epsilon_data = g.train(session, graph, saver, batch_data, epsilon_data)
