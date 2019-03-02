# This is necessary to find the main code
import random
import sys

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import math
import numpy as np
import queue as q

from sensed_world import SensedWorld

entities = ['wall', 'hero', 'enemy', 'bomb', 'explosion', 'monster', 'exit']
actions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'bomb'] # 'wait']

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


    def do(self, wrld):

        # Obtain the state of the current world
        world_cpy = SensedWorld.from_world(wrld)

        my_x = None
        my_y = None
        # Find self in world
        for i, character_list in world_cpy.characters.items():
            for character in character_list:
                if character == world_cpy.me(self):  # Found current player
                    my_x = character.x
                    my_y = character.y

        exit_x = None
        exit_y = None
        # Find exit
        if world_cpy.exitcell is not None:
            exit_x = world_cpy.exitcell[0]
            exit_y = world_cpy.exitcell[1]

        aStarPath = astar(world_cpy, (my_x, my_y), (exit_x, exit_y))

        if len(aStarPath) > 1:
            pos = aStarPath[1]
            delta_x = pos[0] - my_x
            delta_y = pos[1] - my_y

            new_x = self.x + delta_x
            new_y = self.y + delta_y
            if new_x > (len(world_cpy.grid) - 1) or new_x < 0 or new_y > (len(world_cpy.grid[len(world_cpy.grid)-1]) -1) or new_y < 0:
                return
            elif world_cpy.wall_at(new_x, new_y):
                self.place_bomb()
            else:
                self.move(delta_x, delta_y)

            return
        else:
            return


        return













        # TODO Update the Q_Table based on previous state-action-reward
        

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
            self.Q_table[''.join(str(e) for e in state.to_vector())] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
            self.epsilon -= 0.000005
        print("e: " + str(self.epsilon))
        if self.alpha > 0.0000005:
            self.alpha -= 0.00000001
        print("a: " + str(self.alpha))

        if (self.my_x != self.exit_x and self.my_y != self.exit_y):
            astar(wrld.grid, (self.my_x,self.my_y), (self.exit_x,self.exit_y))

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
        # elif action_selection_string == 'wait':
        #     self.move(0, 0)
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


class Node():
    """Node for A*"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(world, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    grid = world.grid


    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (len(grid[len(grid)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            #if grid[node_position[0]][node_position[1]] != 0:
            #    continue

            # Check if already in closed list
            if Node(current_node, node_position) in closed_list:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Find the distance to the closest monster
            distToMonster = 1000
            for i, monster_list in world.monsters.items():
                for monster in monster_list:
                    #distToMonster = min(distToMonster, abs(child.position[0] - monster.x) + abs(child.position[1] - monster.y))
                    distToMonster = min(distToMonster, ((child.position[0] - monster.x) ** 2) + ((child.position[1] - monster.y) ** 2))

            distToMonsterWeight = (1000) * (1.0/(1 + distToMonster))
            #distToMonsterWeight = distToMonster
            distToExitWeight = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            #distToExitWeight = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])


            bombDangerWeight = 0
            for x in range(child.position[0] - 4, child.position[0] + 4):
                if not x > (len(grid) - 1) and not x < 0:
                    if world.bomb_at(x, child.position[1]):
                        bombDangerWeight = 1000
            for y in range(child.position[1] - 4, child.position[1] + 4):
                if not y > (len(grid[len(grid) - 1]) - 1) and not y < 0:
                    if world.bomb_at(child.position[0], y):
                        bombDangerWeight = 1000

            costOfTile = 0
            if world.wall_at(child.position[0], child.position[1]):
                costOfTile += 100
            else:
                costOfTile += 1

            # Create the f, g, and h values
            child.g = current_node.g + costOfTile
            child.h = distToExitWeight + distToMonsterWeight + bombDangerWeight
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child == open_node and child.g > open_node.g]) > 0:
                continue

            for index_dup, dup in enumerate(open_list):
                if dup == child:
                    open_list.pop(index_dup)

            # Add the child to the open list
            open_list.append(child)


    return {}