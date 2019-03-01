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

    def __init__(self):
        self.previous_world = None
        self.ep_reward = 0

    def do(self, wrld):

        graph = build_graph()
        saver = tf.train.Saver(max_to_keep=5)


        # TODO Create a Rewards Table and update the previous reward


        # TODO Update the Q_Table based on previous state-action-reward

        # Obtain the state of the current world
        world_cpy = SensedWorld.from_world(wrld)
        if self.previous_world:
            old_score = self.previous_world.scores[self.name]
            new_score = world_cpy.scores[self.name]
            reward = new_score - old_score
            self.ep_reward += reward

        self.previous_world = world_cpy
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
        if action_selection_string == 'N':
            self.move(0, -1)
        elif action_selection_string == 'NE':
            self.move(1, -1)
        elif action_selection_string == 'NW':
            self.move(-1, -1)
        elif action_selection_string == 'S':
            self.move(0, 1)
        elif action_selection_string == 'SE':
            self.move(1, 1)
        elif action_selection_string == 'SW':
            self.move(-1, 1)
        elif action_selection_string == 'E':
            self.move(1, 0)
        elif action_selection_string == 'W':
            self.move(-1, 0)
        elif action_selection_string == 'wait':
            self.move(0, 0)
        elif action_selection_string == 'bomb':
            self.place_bomb()

        # Update the persistent variables to hold information on this action selection
        # Reward is unknown until action is completed ie start of this function
        global previous_state
        previous_state = grid
        global previous_action
        previous_action = action_selection
        global previous_reward
        previous_reward = reward

        # Profiling
        memory_usage = sys.getsizeof(Q_Table)

        pass


# =============================
#   TFLearn Deep Q Network
# =============================
def build_dqn():
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


def train(session, graph_ops, saver):
    """
    Train a model.
    """

    #summary_ops = build_summaries()  # TODO
    #summary_op = summary_ops[-1]  # TODO

    # Initialize variables
    session.run(tf.initialize_all_variables())
    #writer = writer_summary(summary_dir + "/qlearning", session.graph)  # TODO

    # Initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    #TODO Move the thread here

############################################################################## Interior of thread

    ################################## TODO Persistance of this scope per episode?

    # Unpack the graph operations
    shared_inputs = graph_ops["shared_inputs"]
    q_values = graph_ops["q_values"]
    shared_target_inputs = graph_ops["shared_target_inputs"]
    target_q_values = graph_ops["target_q_values"]
    reset_target_network_params = graph_ops["reset_target_network_params"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    # Initialize network gradients
    state_batch = []
    action_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()  # TODO make the function for epsilon
    initial_epsilon = 1.0
    epsilon = 1.0

    # Get initial game observation
    # TODO get current world state
    world_state = None

    # Set up per-episode counters
    ep_reward = 0
    episode_ave_max_q = 0
    ep_t = 0

    t = 0

    ###################################

    # Forward the deep q network, get Q(s,a) values
    readout_t = q_values.eval(session=session, feed_dict={shared_inputs: [world_state]})  # TODO use world grid

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

    # TODO Update world as if step function
    statePrime = None
    reward = None

    # Accumulate gradients
    readout_j1 = target_q_values.eval(session=session, feed_dict={shared_target_inputs: [statePrime]})

    # Scale rewards to between -1 and 1
    clipped_r_t = np.clip(reward, -1, 1)
    y_batch.append(clipped_r_t + gamma * np.max(readout_j1))  # Use Bellman

    action_batch.append(action_taken)
    state_batch.append(world_state)

    # Update the state and counters
    world_state = statePrime

    ep_t += 1
    ep_reward += reward
    episode_ave_max_q += np.max(readout_t)

    # Save the model
    if t % checkpoint_interval == 0:
        saver.save(session, "qlearning.ckpt", global_step=t)

###############################################################################

    # Show the agents training and write summary statistics
    #last_summary_time = 0
    #while True:
    #    if show_training:
    #        for env in envs:
    #            env.render()
    #    now = time.time()
    #    if now - last_summary_time > summary_interval:
    #        summary_str = session.run(summary_op)
    #        writer.add_summary(summary_str, float(T))
    #        last_summary_time = now
    #for t in actor_learner_threads:
    #    t.join()