# This is necessary to find the main code
import random
import sys

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import math
import numpy as np

from sensed_world import SensedWorld

entities = ['wall', 'hero', 'enemy', 'bomb', 'explosion', 'monster', 'exit']
actions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'bomb', 'wait']

# TODO Save and Load Q_Table from file for persistence (optionally human readable ie JSON)
Q_Table = []

class WorldState:
    def __init__(self):
        self.my_x = 0
        self.my_y = 0
        self.exit_x = 0
        self.exit_y = 0
        # list of bombs
        self.bomb_x = []
        self.bomb_y = []
        # list of explosions
        self.exp_x = []
        self.exp_y = []
        # closest monster
        self.c_monst_x = math.inf
        self.c_monst_y = math.inf
        # list of monsters
        self.monst_x = []
        self.monst_y = []
    
    # returns a tuple consisting of the euclidean distance to the point, the x distance, and the y distance
    def dist(self, other_x, other_y):
        dx = other_x - self.my_x
        dy = other_y - self.my_y
        dist = math.sqrt(dx**2 + dy**2)
        return (dist, dx, dy)
        
    def manhat_dist(self, other_x, other_y):
        dx = other_x - self.my_x
        dy = other_y - self.my_y
        return abs(dx) + abs(dy)
    
    def bomb_danger_x(self):
        for i in range(len(self.bomb_x)):
            if self.my_x == self.bomb_x[i]:
                return 1
        return 0
        
    def bomb_danger_y(self):
        for i in range(len(self.bomb_x)):
            if self.my_y == self.bomb_y[i]:
                return 1
        return 0
        
    # returns a vector of whether each move is safe
    def all_move_danger(self):
        moves = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    moves.append(self.move_danger(i, j))
        return moves
        
    # checks if a single move is safe
    def move_danger(self, dx, dy):
        for i in range(len(self.exp_x)):
            if self.my_x + dx == self.exp_x[i] and self.my_y + dy == self.exp_y[i]:
                return 1
        for i in range(len(self.monst_x)):
            if self.my_x + dx == self.monst_x[i] and self.my_y + dy == self.monst_y[i]:
                return 1
        return 0
        
    def to_vector(self):
        vect_form = []
        exit_dist = self.dist(self.exit_x, self.exit_y)
        vect_form.append(int(exit_dist[0]))
        vect_form.append(exit_dist[1])
        vect_form.append(exit_dist[2])
        vect_form.append(self.manhat_dist(self.exit_x, self.exit_y))
        if self.c_monst_x != math.inf:
            monst_dist = self.dist(self.c_monst_x, self.c_monst_y)
            vect_form.append(int(monst_dist[0]))
            vect_form.append(monst_dist[1])
            vect_form.append(monst_dist[2])
            vect_form.append(self.manhat_dist(self.c_monst_x, self.c_monst_y))
        else:
            vect_form.append(27)
            vect_form.append(10)
            vect_form.append(25)
            vect_form.append(35)
        vect_form.append(self.bomb_danger_x())
        vect_form.append(self.bomb_danger_y())
        vect_form.append(sum(self.all_move_danger()))
        return vect_form
        

class TestCharacter(CharacterEntity):

    def __init__(self, name, avatar, x, y, alpha, epsilon, Q_table, weights, actions_taken):
        CharacterEntity.__init__(self, name, avatar, x, y)
        self.alpha = alpha
        self.gamma = 0.95
        self.epsilon = epsilon
        self.Q_table = Q_table
        self.weights = weights
        self.actions_taken = actions_taken
        self.Q_prime = dict()
        self.prev_world = None
        self.prev_state = None
        self.prev_action = 9
        self.prev_score = 0

    def do(self, wrld):

        # TODO Update the Q_Table based on previous state-action-reward
        
        # Obtain the state of the current world
        world_cpy = SensedWorld.from_world(wrld)
        Q_sa = 0
        reward = 0
        new_score = None
        if self.prev_world is not None:
            old_score = self.prev_world.scores[self.name]
            new_score = world_cpy.scores[self.name]
            reward = new_score - old_score - (5.0*self.prev_state.dist(self.prev_state.exit_x, self.prev_state.exit_y)[0])
            Q_sa = np.dot(self.weights, np.array(self.prev_state.to_vector())).item(0)
            print("Q_sa: " + str(Q_sa))
            print("weights: " + str(self.weights))
            print("f_vec: " + str(np.array(self.prev_state.to_vector())))
            self.Q_table[''.join(str(e) for e in self.prev_state.to_vector())][self.prev_action] = Q_sa

        # Save relevant state details to class
        state = WorldState()

        # Find self in world
        for i, character_list in world_cpy.characters.items():
            for character in character_list:
                if character == world_cpy.me(self):  # Found current player
                    state.my_x = character.x
                    state.my_y = character.y

        # Find bombs in world
        for bomb in world_cpy.bombs.items():
            state.bomb_x.append(bomb[1].x)
            state.bomb_y.append(bomb[1].y)
            
        # Find explosions in world
        for explosion in world_cpy.explosions.items():
            state.exp_x.append(explosion[1].x)
            state.exp_y.append(explosion[1].y)
            
        # Find monsters in world
        for i, monster_list in world_cpy.monsters.items():
            for monster in monster_list:
                state.monst_x.append(monster.x)
                state.monst_y.append(monster.y)
                closest_monst = state.dist(state.c_monst_x, state.c_monst_y)
                new_monst = state.dist(monster.x, monster.y)
                if new_monst[0] < closest_monst[0]:
                    state.c_monst_x = monster.x
                    state.c_monst_y = monster.y

        # Find exit
        if world_cpy.exitcell is not None:
            state.exit_x = world_cpy.exitcell[0]
            state.exit_y = world_cpy.exitcell[1]
        
        # update Q table
        if ''.join(str(e) for e in state.to_vector()) not in list(self.Q_table.keys()):
            self.Q_table[''.join(str(e) for e in state.to_vector())] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Q_prime = self.Q_table[''.join(str(e) for e in state.to_vector())]
            
        
        self.update_weights(state.to_vector(), reward, Q_sa)
        
        # Using epsilon greedy formula, choose an action based on the obtained action score pairs
        x = random.random()
        if x < self.epsilon:
            action_selection = random.randint(0, len(actions) - 1)
        else:
            max_q = -1.0*math.inf
            action_selection = None
            for i in range(len(self.Q_prime)):
                if (self.Q_prime[i] / 5.0 + (5.0 * self.actions_taken[i])) > max_q:
                    max_q = self.Q_prime[i]
                    action_selection = i
            if action_selection is None:
                print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n" +
                        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n" +
                        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
                action_selection = random.randint(0, len(actions) - 1)
                
        if self.epsilon > 0:
            self.epsilon -= 0.00001
        print("e: " + str(self.epsilon))
        if self.alpha > 0.0000005:
            self.alpha -= 0.00000001
        print("a: " + str(self.alpha))

        # Parse action selection and perform action
        action_selection_string = actions[action_selection]
        self.actions_taken[action_selection] += 1
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
        self.prev_world = world_cpy
        self.prev_state = state
        self.prev_action = action_selection
        if new_score is not None:
            self.prev_score = new_score

        # Profiling
        #memory_usage = sys.getsizeof(self.Q_table)

        pass
    
    # taken directly from approximate Q slides
    def update_weights(self, fs, r, Q_sa):
        delta = (r + self.gamma * max(self.Q_prime)) - Q_sa
        for i in range(len(self.weights)):
            self.weights[i] += self.alpha * delta * fs[i]
